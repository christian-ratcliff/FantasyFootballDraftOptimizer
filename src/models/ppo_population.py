"""
Population-based Training for Fantasy Football Draft Optimization

This module implements population-based training with self-play for fantasy football drafting,
allowing a collection of PPO agents to compete, evolve, and improve over time.
"""
import os
import copy
import random
import numpy as np
import logging
import time
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional, Union
import tensorflow as tf
from keras import backend as K

from src.models.ppo_drafter import PPODrafter, DraftState
from src.models.hierarchical_ppo_drafter import HierarchicalPPODrafter
from src.models.draft_simulator import DraftSimulator, Team, Player
from src.models.season_simulator import SeasonSimulator, SeasonEvaluator

logger = logging.getLogger(__name__)

class PPOPopulation:
    """
    Manages a population of PPO agents for fantasy football draft optimization
    using population-based training with self-play competitions.
    """
    
    def __init__(self, 
                 population_size: int = 5,
                 state_dim: int = 100, 
                 action_feature_dim: int = 50, 
                 action_dim: int = 256,
                 hierarchical_ratio: float = 0.5,
                 evolution_interval: int = 50,
                 tournament_size: int = 3,
                 mutation_rate: float = 0.1,
                 mutation_strength: float = 0.2,
                 elitism_count: int = 1,
                 output_dir: Optional[str] = None,
                 use_top_n_features: int = 0,
                 enable_crossover: bool = True,
                 curriculum_enabled: bool = True,
                 opponent_modeling_enabled: bool = True):
        """
        Initialize the PPO population
        
        Parameters:
        -----------
        population_size : int
            Number of agents in the population
        state_dim : int
            Dimension of state vector
        action_feature_dim : int
            Dimension of action feature vector
        action_dim : int
            Maximum number of actions (players to choose from)
        hierarchical_ratio : float
            Ratio of hierarchical PPO agents in the population (0.0 to 1.0)
        evolution_interval : int
            Number of episodes between evolution events
        tournament_size : int
            Number of agents participating in each tournament
        mutation_rate : float
            Probability of a parameter being mutated
        mutation_strength : float
            Scale of mutations
        elitism_count : int
            Number of top agents to preserve unchanged
        output_dir : str, optional
            Directory to save models and results
        use_top_n_features : int
            Number of top features to use from projection models  
        enable_crossover : bool
            Whether to enable crossover (genetic recombination)
        curriculum_enabled : bool
            Whether to use curriculum learning for agents
        opponent_modeling_enabled : bool
            Whether to use opponent modeling for agents
        """
        self.population_size = population_size
        self.state_dim = state_dim
        self.action_feature_dim = action_feature_dim
        self.action_dim = action_dim
        self.hierarchical_ratio = hierarchical_ratio
        self.evolution_interval = evolution_interval
        self.tournament_size = min(tournament_size, population_size)
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elitism_count = min(elitism_count, population_size)
        self.output_dir = output_dir
        self.use_top_n_features = use_top_n_features
        self.enable_crossover = enable_crossover
        self.curriculum_enabled = curriculum_enabled
        self.opponent_modeling_enabled = opponent_modeling_enabled
        
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.population_dir = os.path.join(output_dir, 'population')
            os.makedirs(self.population_dir, exist_ok=True)
            
        # Initialize the population with appropriate types
        self.population = []
        self.initialize_population()
        
        # Track population metrics
        self.generation = 0
        self.fitness_history = []
        self.win_rates = []
        self.championship_rates = []
        self.avg_rewards = []
        self.diversity_metrics = []
        self.tournament_results = []
        
        # Keep track of best agent
        self.best_agent_idx = 0
        self.best_fitness = -float('inf')
        
        # Track creation time for this generation
        self.generation_creation_time = time.time()
        
    def initialize_population(self):
        """Initialize the population with a mix of standard and hierarchical PPO agents"""
        self.population = []
        
        # Calculate how many hierarchical agents to create
        hierarchical_count = int(self.population_size * self.hierarchical_ratio)
        standard_count = self.population_size - hierarchical_count
        
        logger.info(f"Initializing population with {standard_count} standard and {hierarchical_count} hierarchical agents")
        
        # Create standard PPO agents
        for i in range(standard_count):
            # Initialize with slight variations in hyperparameters
            lr_variation = random.uniform(0.8, 1.2)
            entropy_variation = random.uniform(0.8, 1.2)
            
            agent = PPODrafter(
                state_dim=self.state_dim,
                action_feature_dim=self.action_feature_dim,
                action_dim=self.action_dim,
                lr_actor=0.0003 * lr_variation,
                lr_critic=0.0003 * lr_variation,
                gamma=0.96,
                entropy_coef=0.01 * entropy_variation,
                use_top_n_features=self.use_top_n_features,
                curriculum_enabled=self.curriculum_enabled,
                opponent_modeling_enabled=self.opponent_modeling_enabled
            )
            
            # Set agent metadata
            agent.agent_id = f"standard_{i}"
            agent.agent_type = "standard"
            agent.creation_time = time.time()
            agent.fitness = 0.0
            agent.lifetime_episodes = 0
            agent.parent_ids = []  # Track lineage
            agent.generation = 0
            agent.hyperparams = {
                "lr_actor": 0.0003 * lr_variation,
                "lr_critic": 0.0003 * lr_variation,
                "gamma": 0.96,
                "entropy_coef": 0.01 * entropy_variation,
            }
            
            self.population.append(agent)
        
        # Create hierarchical PPO agents
        for i in range(hierarchical_count):
            # Initialize with slight variations in hyperparameters
            meta_lr_variation = random.uniform(0.8, 1.2)
            sub_lr_variation = random.uniform(0.8, 1.2)
            entropy_variation = random.uniform(0.8, 1.2)
            
            agent = HierarchicalPPODrafter(
                state_dim=self.state_dim,
                action_feature_dim=self.action_feature_dim,
                action_dim=self.action_dim,
                lr_meta=0.0003 * meta_lr_variation,
                lr_sub=0.0003 * sub_lr_variation,
                lr_critic=0.0003 * meta_lr_variation,
                gamma=0.96,
                entropy_coef=0.01 * entropy_variation,
                use_top_n_features=self.use_top_n_features,
                curriculum_enabled=self.curriculum_enabled,
                opponent_modeling_enabled=self.opponent_modeling_enabled
            )
            
            # Set agent metadata
            agent.agent_id = f"hierarchical_{i}"
            agent.agent_type = "hierarchical"
            agent.creation_time = time.time()
            agent.fitness = 0.0
            agent.lifetime_episodes = 0
            agent.parent_ids = []  # Track lineage
            agent.generation = 0
            agent.hyperparams = {
                "lr_meta": 0.0003 * meta_lr_variation,
                "lr_sub": 0.0003 * sub_lr_variation,
                "lr_critic": 0.0003 * meta_lr_variation,
                "gamma": 0.96,
                "entropy_coef": 0.01 * entropy_variation,
            }
            
            self.population.append(agent)
        
        logger.info(f"Population initialized with {len(self.population)} agents")
    
    def train_population(self, 
                        draft_simulator: DraftSimulator, 
                        season_simulator: SeasonSimulator,
                        num_episodes: int = 200, 
                        eval_interval: int = 10,
                        save_interval: int = 50):
        """
        Train all agents in the population with periodic evolution
        
        Parameters:
        -----------
        draft_simulator : DraftSimulator
            Draft simulator instance
        season_simulator : SeasonSimulator
            Season simulator instance
        num_episodes : int
            Number of episodes to train for
        eval_interval : int
            Number of episodes between evaluations
        save_interval : int
            Number of episodes between saving models
        
        Returns:
        --------
        dict
            Training results
        """
        start_time = time.time()
        logger.info(f"Starting population training for {num_episodes} episodes")
        
        # Track overall population results
        population_results = {
            "generations": self.generation,
            "fitness_history": [],
            "win_rates": [],
            "diversity": [],
            "best_agent": None,
            "training_time": 0
        }
        
        # Train each agent for the specified number of episodes
        for episode in range(1, num_episodes + 1):
            episode_start_time = time.time()
            logger.info(f"Episode {episode}/{num_episodes}")
            
            # Train each agent for one episode
            episode_results = []
            for idx, agent in enumerate(self.population):
                logger.info(f"Training agent {idx} ({agent.agent_id})")
                
                # Create fresh copies of simulators for this agent
                agent_draft_sim = self._create_fresh_simulator(draft_simulator, draft_simulator.teams)
                agent_season_sim = self._create_fresh_simulator(season_simulator, draft_simulator.teams)
                
                # Train for one episode
                result = self._train_single_episode(
                    agent=agent,
                    draft_simulator=agent_draft_sim,
                    season_simulator=agent_season_sim,
                    episode=episode
                )
                
                episode_results.append(result)
                
                # Update agent metadata
                agent.lifetime_episodes += 1
            
            # Log episode timing
            episode_time = time.time() - episode_start_time
            logger.info(f"Episode {episode} completed in {episode_time:.2f} seconds")
            
            # Run evaluation if needed
            if episode % eval_interval == 0:
                logger.info(f"Evaluating population at episode {episode}")
                self.evaluate_population(draft_simulator, season_simulator)
                
                # Save evaluation metrics
                population_results["fitness_history"].append({
                    "episode": episode,
                    "fitness": [agent.fitness for agent in self.population],
                    "best_fitness": self.best_fitness,
                    "best_agent_id": self.population[self.best_agent_idx].agent_id
                })
                
                # Calculate and log population diversity
                diversity = self.calculate_population_diversity()
                population_results["diversity"].append({
                    "episode": episode,
                    "weight_diversity": diversity["weight_diversity"],
                    "hyperparameter_diversity": diversity["hyperparameter_diversity"]
                })
                logger.info(f"Population diversity: Weight={diversity['weight_diversity']:.4f}, Hyperparameter={diversity['hyperparameter_diversity']:.4f}")
                
                # Update visualization
                self.visualize_population_metrics(output_path=os.path.join(self.output_dir, 'population_metrics.png'))
            
            # Save models if needed
            if self.output_dir and episode % save_interval == 0:
                logger.info(f"Saving population at episode {episode}")
                self.save_population(os.path.join(self.population_dir, f"population_ep{episode}"))
            
            # Perform evolution if needed
            if episode % self.evolution_interval == 0:
                logger.info(f"Evolving population at episode {episode}")
                self.evolve_population()
                self.generation += 1
                
                # Update creation time for this generation
                self.generation_creation_time = time.time()
                
                # Log new generation info
                logger.info(f"Generation {self.generation} created")
                for idx, agent in enumerate(self.population):
                    logger.info(f"  Agent {idx}: {agent.agent_id}, Type: {agent.agent_type}, Parents: {agent.parent_ids}")
        
        # Final evaluation
        logger.info("Performing final population evaluation")
        self.evaluate_population(draft_simulator, season_simulator)
        
        # Save final population
        if self.output_dir:
            logger.info("Saving final population")
            self.save_population(os.path.join(self.population_dir, "population_final"))
            
            # Save the best agent separately
            best_agent = self.population[self.best_agent_idx]
            best_agent_path = os.path.join(self.output_dir, "best_agent")
            if isinstance(best_agent, HierarchicalPPODrafter):
                best_agent.save_model(best_agent_path)
            else:
                best_agent.save_model(best_agent_path)
            
            # Save population metrics
            with open(os.path.join(self.output_dir, "population_results.json"), "w") as f:
                # Make sure everything is JSON serializable
                serializable_results = self._make_serializable(population_results)
                json.dump(serializable_results, f, indent=4)
        
        # Calculate final metrics
        training_time = time.time() - start_time
        logger.info(f"Population training completed in {training_time:.2f} seconds")
        
        # Return the best agent and results
        population_results["best_agent"] = self.population[self.best_agent_idx]
        population_results["training_time"] = training_time
        
        return population_results
    
    def _train_single_episode(self, agent, draft_simulator, season_simulator, episode):
        """
        Train a single agent for one episode
        
        Parameters:
        -----------
        agent : PPODrafter or HierarchicalPPODrafter
            Agent to train
        draft_simulator : DraftSimulator
            Draft simulator instance
        season_simulator : SeasonSimulator
            Season simulator instance
        episode : int
            Current episode number
            
        Returns:
        --------
        dict
            Episode results
        """
        # Assign a random draft position
        draft_position = random.randint(1, draft_simulator.league_size)
        
        # Set up the team with PPO strategy
        rl_team = None
        for team in draft_simulator.teams:
            if team.draft_position == draft_position:
                team.strategy = "PPO"
                rl_team = team
                break
        
        if not rl_team:
            rl_team = random.choice(draft_simulator.teams)
            rl_team.strategy = "PPO"
        
        # Make sure only one team is using PPO
        for team in draft_simulator.teams:
            if team != rl_team and team.strategy == "PPO":
                team.strategy = "VBD"  # Default fallback
        
        # Train for one episode
        episode_results = {}
        
        # Use the appropriate training method based on agent type
        if isinstance(agent, HierarchicalPPODrafter):
            # Train hierarchical agent
            results = agent.train(
                draft_simulator=draft_simulator,
                season_simulator=season_simulator,
                num_episodes=1,
                eval_interval=1,
                save_interval=1000,  # Set high to avoid saving
                save_path=None  # No saving during regular training
            )
            episode_results = results
        else:
            # Train standard PPO agent
            results = agent.train(
                draft_simulator=draft_simulator,
                season_simulator=season_simulator,
                num_episodes=1,
                eval_interval=1,
                save_interval=1000,  # Set high to avoid saving
                save_path=None  # No saving during regular training
            )
            episode_results = results
        
        return episode_results
    
    def evaluate_population(self, draft_simulator, season_simulator):
        """
        Evaluate all agents in the population using tournaments
        
        Parameters:
        -----------
        draft_simulator : DraftSimulator
            Draft simulator instance
        season_simulator : SeasonSimulator
            Season simulator instance
        """
        # Number of tournament rounds
        num_tournaments = max(3, self.population_size // 2)
        
        # Reset fitness scores
        for agent in self.population:
            agent.tournament_wins = 0
            agent.tournament_entries = 0
            agent.points_total = 0
            agent.championships = 0
        
        # Run tournaments
        for t in range(num_tournaments):
            logger.info(f"Running tournament {t+1}/{num_tournaments}")
            
            # Select random participants for this tournament
            tournament_size = min(self.tournament_size, len(self.population))
            participants = random.sample(self.population, tournament_size)
            
            # Create a fresh draft simulator for this tournament
            tournament_draft_sim = self._create_fresh_simulator(draft_simulator, draft_simulator.teams)
            tournament_season_sim = self._create_fresh_simulator(season_simulator, draft_simulator.teams)
            
            # Run the tournament
            results = self._run_tournament(
                participants=participants,
                draft_simulator=tournament_draft_sim, 
                season_simulator=tournament_season_sim
            )
            
            # Record results
            for i, agent in enumerate(participants):
                agent.tournament_entries += 1
                if i == results["winner_idx"]:
                    agent.tournament_wins += 1
                
                # Record points from their team
                agent_team = results["team_results"][i]
                agent.points_total += agent_team["points"]
                
                # Record championships
                if agent_team.get("championship", False):
                    agent.championships += 1
        
        # Calculate fitness scores
        for agent in self.population:
            # Avoid division by zero
            tournament_win_rate = agent.tournament_wins / max(1, agent.tournament_entries)
            avg_points = agent.points_total / max(1, agent.tournament_entries)
            championship_rate = agent.championships / max(1, agent.tournament_entries)
            
            # Fitness formula - weighted combination of metrics
            agent.fitness = (
                tournament_win_rate * 5.0 +
                avg_points * 0.05 +
                championship_rate * 10.0
            )
            
            logger.info(f"Agent {agent.agent_id}: Fitness={agent.fitness:.2f}, Win Rate={tournament_win_rate:.2f}, Avg Points={avg_points:.1f}")
        
        # Update best agent
        current_best_idx = max(range(len(self.population)), key=lambda i: self.population[i].fitness)
        if self.population[current_best_idx].fitness > self.best_fitness:
            self.best_agent_idx = current_best_idx
            self.best_fitness = self.population[current_best_idx].fitness
            logger.info(f"New best agent: {self.population[current_best_idx].agent_id} with fitness {self.best_fitness:.2f}")
        
        # Record fitness history
        self.fitness_history.append([agent.fitness for agent in self.population])
    
    def _run_tournament(self, participants, draft_simulator, season_simulator):
        """
        Run a tournament between multiple agents
        
        Parameters:
        -----------
        participants : list
            List of agents participating in the tournament
        draft_simulator : DraftSimulator
            Draft simulator instance
        season_simulator : SeasonSimulator
            Season simulator instance
                
        Returns:
        --------
        dict
            Tournament results
        """
        # Create a copy of the simulator for this tournament
        tournament_draft_sim = draft_simulator
        
        # Each agent gets a team - assign in random order
        agent_team_indices = []
        for i, agent in enumerate(participants):
            # Find a team to use for this agent
            draft_position = i + 1  # Assign draft positions sequentially
            
            team_idx = None
            for j, team in enumerate(tournament_draft_sim.teams):
                if team.draft_position == draft_position:
                    team_idx = j
                    team.strategy = "PPO"  # Mark as PPO team
                    agent_team_indices.append(j)
                    break
            
            if team_idx is None:
                # If no matching position, just take the next available team
                for j, team in enumerate(tournament_draft_sim.teams):
                    if j not in agent_team_indices:
                        team_idx = j
                        team.strategy = "PPO"  # Mark as PPO team
                        agent_team_indices.append(j)
                        break
            
            # Assign agent to this team
            tournament_draft_sim.teams[team_idx].agent = agent
        
        # Ensure other teams use regular strategies
        for i, team in enumerate(tournament_draft_sim.teams):
            if i not in agent_team_indices:
                # Assign a non-PPO strategy
                team.strategy = random.choice(["VBD", "ESPN", "ZeroRB", "HeroRB", "TwoRB", "BestAvailable"])
        
        # Run the draft
        teams, draft_history = tournament_draft_sim.run_draft()
        
        # Ensure we have the correct number of playoff teams
        if season_simulator.num_playoff_teams < 4:
            # The minimum needed for a proper semi-final is 4 teams
            season_simulator.num_playoff_teams = 4
            logger.info(f"Adjusted playoff teams to {season_simulator.num_playoff_teams} to ensure proper bracket")
        
        # Simulate the season
        season_results = season_simulator.simulate_season()
        
        # Evaluate the results
        evaluation = SeasonEvaluator(teams, season_results)
        
        # Determine the winner
        winner_idx = 0
        best_rank = float('inf')
        
        # Track results for each agent's team
        team_results = []
        
        for i, team_idx in enumerate(agent_team_indices):
            team = tournament_draft_sim.teams[team_idx]
            
            # Find team in the standings
            team_standing = None
            for standing in season_results["standings"]:
                if standing["team"] == team.name:
                    team_standing = standing
                    break
            
            if team_standing:
                rank = team_standing["rank"]
                points = team_standing["points_for"]
                
                # Check if this is the best performing agent
                if rank < best_rank:
                    best_rank = rank
                    winner_idx = i
                
                # Record if they were the champion
                is_champion = (season_results.get("playoffs", {}).get("champion") == team.name)
                
                team_results.append({
                    "rank": rank,
                    "points": points,
                    "championship": is_champion
                })
            else:
                # If standing not found, use default values
                team_results.append({
                    "rank": tournament_draft_sim.league_size,
                    "points": 0,
                    "championship": False
                })
        
        # Compile tournament results
        results = {
            "winner_idx": winner_idx,
            "team_results": team_results,
            "draft_history": draft_history,
            "season_results": season_results
        }
        
        return results
    
    def evolve_population(self):
        """
        Evolve the population using tournament selection, crossover, and mutation
        """
        # Sort population by fitness
        sorted_indices = sorted(range(len(self.population)), 
                               key=lambda i: self.population[i].fitness,
                               reverse=True)
        
        # Create new population
        new_population = []
        
        # Elitism: Keep top performers unchanged
        for i in range(self.elitism_count):
            elite_idx = sorted_indices[i]
            elite_agent = self.population[elite_idx]
            
            # Create a deep copy to preserve the agent
            if isinstance(elite_agent, HierarchicalPPODrafter):
                elite_copy = self._clone_hierarchical_agent(elite_agent)
            else:
                elite_copy = self._clone_standard_agent(elite_agent)
            
            # Update agent metadata but preserve fitness
            elite_copy.agent_id = f"{elite_agent.agent_type}_{self.generation}_{i}_elite"
            elite_copy.parent_ids = [elite_agent.agent_id]
            elite_copy.generation = self.generation
            
            # Add to new population
            new_population.append(elite_copy)
            
            logger.info(f"Elite selection: {elite_agent.agent_id} -> {elite_copy.agent_id}")
        
        # Fill the rest with offspring from tournament selection, crossover, and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            
            if self.enable_crossover and random.random() < 0.7:  # 70% chance of crossover
                # Select another parent for crossover
                parent2 = self._tournament_selection()
                
                # Ensure we don't crossover the same agent
                while parent2 == parent1:
                    parent2 = self._tournament_selection()
                
                # Create child through crossover and mutation
                child = self._crossover_and_mutate(parent1, parent2)
                
                # Update metadata
                child.agent_id = f"{child.agent_type}_{self.generation}_{len(new_population)}_crossover"
                child.parent_ids = [parent1.agent_id, parent2.agent_id]
                child.generation = self.generation
                
                logger.info(f"Crossover: {parent1.agent_id} + {parent2.agent_id} -> {child.agent_id}")
            else:
                # Just mutate the selected parent
                child = self._mutate(parent1)
                
                # Update metadata
                child.agent_id = f"{child.agent_type}_{self.generation}_{len(new_population)}_mutated"
                child.parent_ids = [parent1.agent_id]
                child.generation = self.generation
                
                logger.info(f"Mutation: {parent1.agent_id} -> {child.agent_id}")
            
            # Reset fitness of child
            child.fitness = 0.0
            child.lifetime_episodes = 0
            
            # Add to new population
            new_population.append(child)
        
        # Replace old population with new one
        self.population = new_population
    
    def _tournament_selection(self):
        """
        Select an agent using tournament selection
        
        Returns:
        --------
        Agent
            Selected agent
        """
        # Select random participants
        tournament_size = min(3, len(self.population))
        participants = random.sample(self.population, tournament_size)
        
        # Return the fittest
        return max(participants, key=lambda agent: agent.fitness)
    
    def _crossover_and_mutate(self, parent1, parent2):
        """
        Create a child agent through crossover and mutation
        
        Parameters:
        -----------
        parent1 : PPODrafter or HierarchicalPPODrafter
            First parent
        parent2 : PPODrafter or HierarchicalPPODrafter
            Second parent
                
        Returns:
        --------
        PPODrafter or HierarchicalPPODrafter
            Child agent
        """
        # Check if parents are the same type
        if type(parent1) != type(parent2):
            # Handle mixed coupling - pick one type randomly
            if random.random() < 0.5:
                # If different types, clone parent1 and just mutate
                return self._mutate(parent1)
            else:
                # If different types, clone parent2 and just mutate
                return self._mutate(parent2)
        
        # Create the child by cloning parent1 (structure will be the same)
        if isinstance(parent1, HierarchicalPPODrafter):
            child = self._clone_hierarchical_agent(parent1)
        else:
            child = self._clone_standard_agent(parent1)
        
        # Perform crossover of hyperparameters
        for param in child.hyperparams:
            if param in parent2.hyperparams:
                # 50% chance of taking either parent's value
                if random.random() < 0.5:
                    child.hyperparams[param] = parent2.hyperparams[param]
        
        # Apply hyperparameters to the child agent
        if isinstance(child, HierarchicalPPODrafter):
            # Apply hyperparameters to hierarchical agent
            if "lr_meta" in child.hyperparams:
                K.set_value(child.meta_policy.optimizer.learning_rate, child.hyperparams["lr_meta"])
            
            if "lr_sub" in child.hyperparams:
                for position in child.sub_policies:
                    K.set_value(child.sub_policies[position].optimizer.learning_rate, child.hyperparams["lr_sub"])
            
            if "lr_critic" in child.hyperparams:
                K.set_value(child.critic.optimizer.learning_rate, child.hyperparams["lr_critic"])
            
            if "entropy_coef" in child.hyperparams:
                child.entropy_coef = child.hyperparams["entropy_coef"]
            
            if "gamma" in child.hyperparams:
                child.gamma = child.hyperparams["gamma"]
                
        else:
            # Apply hyperparameters to standard agent
            if "lr_actor" in child.hyperparams:
                K.set_value(child.actor.optimizer.learning_rate, child.hyperparams["lr_actor"])
            
            if "lr_critic" in child.hyperparams:
                K.set_value(child.critic.optimizer.learning_rate, child.hyperparams["lr_critic"])
            
            if "entropy_coef" in child.hyperparams:
                child.entropy_coef = child.hyperparams["entropy_coef"]
            
            if "gamma" in child.hyperparams:
                child.gamma = child.hyperparams["gamma"]
        
        # Crossover of network weights
        if isinstance(child, HierarchicalPPODrafter):
            # Crossover weights for hierarchical agent
            # Meta policy
            self._crossover_network_weights(child.meta_policy, parent1.meta_policy, parent2.meta_policy)
            
            # Sub policies - only crossover matching positions
            for position in child.sub_policies:
                if position in parent1.sub_policies and position in parent2.sub_policies:
                    self._crossover_network_weights(
                        child.sub_policies[position], 
                        parent1.sub_policies[position], 
                        parent2.sub_policies[position]
                    )
            
            # Critic
            self._crossover_network_weights(child.critic, parent1.critic, parent2.critic)
            
        else:
            # Crossover weights for standard agent
            # Actor
            self._crossover_network_weights(child.actor, parent1.actor, parent2.actor)
            
            # Critic
            self._crossover_network_weights(child.critic, parent1.critic, parent2.critic)
        
        # Apply mutation to the child
        child = self._mutate(child)
        
        return child
    
    def _crossover_network_weights(self, child_network, parent1_network, parent2_network):
        """
        Perform crossover between two parent networks
        
        Parameters:
        -----------
        child_network : tf.keras.Model
            Child network to modify
        parent1_network : tf.keras.Model
            First parent network
        parent2_network : tf.keras.Model
            Second parent network
        """
        # Get the network weights from the child and parents
        child_weights = child_network.get_weights()
        parent1_weights = parent1_network.get_weights()
        parent2_weights = parent2_network.get_weights()
        
        # Make sure shapes match
        if len(child_weights) != len(parent1_weights) or len(child_weights) != len(parent2_weights):
            logger.warning(f"Weight shapes don't match for crossover. Using parent1 weights.")
            child_network.set_weights(parent1_weights)
            return
        
        # Iterate through each layer of weights
        for i in range(len(child_weights)):
            # Get the current layer weights from parents
            p1_weights = parent1_weights[i]
            p2_weights = parent2_weights[i]
            
            # We can only crossover if shapes match
            if p1_weights.shape != p2_weights.shape:
                # If shapes don't match, use parent1's weights
                child_weights[i] = p1_weights
                continue
            
            # Uniform crossover at the layer level: 50% chance to take weights from either parent
            if random.random() < 0.5:
                child_weights[i] = p1_weights
            else:
                child_weights[i] = p2_weights
        
        # Set the new weights in the child network
        child_network.set_weights(child_weights)
    
    def _mutate(self, agent):
        """
        Mutate an agent
        
        Parameters:
        -----------
        agent : PPODrafter or HierarchicalPPODrafter
            Agent to mutate
            
        Returns:
        --------
        PPODrafter or HierarchicalPPODrafter
            Mutated agent
        """
        # Clone the agent
        if isinstance(agent, HierarchicalPPODrafter):
            mutated = self._clone_hierarchical_agent(agent)
        else:
            mutated = self._clone_standard_agent(agent)
        
        # Mutate hyperparameters
        for param in mutated.hyperparams:
            if random.random() < self.mutation_rate:
                current_value = mutated.hyperparams[param]
                # Apply mutation with mutation strength
                mutation_factor = 1.0 + random.uniform(-self.mutation_strength, self.mutation_strength)
                mutated.hyperparams[param] = current_value * mutation_factor
        
        # Apply mutated hyperparameters
        if isinstance(mutated, HierarchicalPPODrafter):
            # Apply hyperparameters to hierarchical agent
            if "lr_meta" in mutated.hyperparams:
                K.set_value(mutated.meta_policy.optimizer.learning_rate, mutated.hyperparams["lr_meta"])
            
            if "lr_sub" in mutated.hyperparams:
                for position in mutated.sub_policies:
                    K.set_value(mutated.sub_policies[position].optimizer.learning_rate, mutated.hyperparams["lr_sub"])
            
            if "lr_critic" in mutated.hyperparams:
                K.set_value(mutated.critic.optimizer.learning_rate, mutated.hyperparams["lr_critic"])
            
            if "entropy_coef" in mutated.hyperparams:
                mutated.entropy_coef = mutated.hyperparams["entropy_coef"]
            
            if "gamma" in mutated.hyperparams:
                mutated.gamma = mutated.hyperparams["gamma"]
                    
        else:
            # Apply hyperparameters to standard agent
            if "lr_actor" in mutated.hyperparams:
                K.set_value(mutated.actor.optimizer.learning_rate, mutated.hyperparams["lr_actor"])
            
            if "lr_critic" in mutated.hyperparams:
                K.set_value(mutated.critic.optimizer.learning_rate, mutated.hyperparams["lr_critic"])
            
            if "entropy_coef" in mutated.hyperparams:
                mutated.entropy_coef = mutated.hyperparams["entropy_coef"]
            
            if "gamma" in mutated.hyperparams:
                mutated.gamma = mutated.hyperparams["gamma"]
        
        # Mutate network weights
        if isinstance(mutated, HierarchicalPPODrafter):
            # Mutate weights for hierarchical agent
            
            # Meta policy
            self._mutate_network_weights(mutated.meta_policy)
            
            # Sub policies
            for position in mutated.sub_policies:
                self._mutate_network_weights(mutated.sub_policies[position])
            
            # Critic
            self._mutate_network_weights(mutated.critic)
            
        else:
            # Mutate weights for standard agent
            
            # Actor
            self._mutate_network_weights(mutated.actor)
            
            # Critic
            self._mutate_network_weights(mutated.critic)
        
        return mutated
    
    def _mutate_network_weights(self, network):
        """
        Mutate the weights of a network
        
        Parameters:
        -----------
        network : tf.keras.Model
            Network to mutate
        """
        # Get the network weights
        weights = network.get_weights()
        
        # Iterate through each layer
        for i in range(len(weights)):
            layer_weights = weights[i]
            
            # Create a mutation mask - only mutate some weights
            mutation_mask = np.random.random(layer_weights.shape) < self.mutation_rate
            
            # Apply noise to the selected weights
            noise = np.random.normal(0, self.mutation_strength, layer_weights.shape)
            layer_weights[mutation_mask] += noise[mutation_mask] * layer_weights[mutation_mask]
            
            # Update the weights
            weights[i] = layer_weights
        
        # Set the new weights
        network.set_weights(weights)
    
    def _clone_standard_agent(self, agent):
        """
        Create a deep copy of a standard PPO agent
        
        Parameters:
        -----------
        agent : PPODrafter
            Agent to clone
            
        Returns:
        --------
        PPODrafter
            Cloned agent
        """
        # Create a new agent with the same configuration
        clone = PPODrafter(
            state_dim=agent.state_dim,
            action_feature_dim=agent.action_feature_dim,
            action_dim=agent.action_dim,
            lr_actor=agent.hyperparams.get("lr_actor", 0.0003),
            lr_critic=agent.hyperparams.get("lr_critic", 0.0003),
            gamma=agent.hyperparams.get("gamma", 0.96),
            entropy_coef=agent.hyperparams.get("entropy_coef", 0.01),
            use_top_n_features=agent.use_top_n_features if hasattr(agent, 'use_top_n_features') else self.use_top_n_features,
            curriculum_enabled=agent.curriculum_enabled if hasattr(agent, 'curriculum_enabled') else self.curriculum_enabled,
            opponent_modeling_enabled=agent.opponent_modeling_enabled if hasattr(agent, 'opponent_modeling_enabled') else self.opponent_modeling_enabled
        )
        
        # Copy network weights
        
        # Create dummy data for initialization
        dummy_state = np.zeros((1, agent.state_dim))
        dummy_action_features = np.zeros((1, agent.action_dim, agent.action_feature_dim))
        
        # Initialize networks by doing a forward pass
        clone.actor([dummy_state, dummy_action_features])
        clone.critic(dummy_state)
        
        # Copy actor weights
        clone.actor.set_weights(agent.actor.get_weights())
        
        # Copy critic weights
        clone.critic.set_weights(agent.critic.get_weights())
        
        # Copy metadata
        clone.hyperparams = copy.deepcopy(agent.hyperparams)
        
        return clone
    
    def _clone_hierarchical_agent(self, agent):
        """
        Create a deep copy of a hierarchical PPO agent
        
        Parameters:
        -----------
        agent : HierarchicalPPODrafter
            Agent to clone
            
        Returns:
        --------
        HierarchicalPPODrafter
            Cloned agent
        """
        # Create a new agent with the same configuration
        clone = HierarchicalPPODrafter(
            state_dim=agent.state_dim,
            action_feature_dim=agent.action_feature_dim,
            action_dim=agent.action_dim,
            lr_meta=agent.hyperparams.get("lr_meta", 0.0003),
            lr_sub=agent.hyperparams.get("lr_sub", 0.0003),
            lr_critic=agent.hyperparams.get("lr_critic", 0.0003),
            gamma=agent.hyperparams.get("gamma", 0.96),
            entropy_coef=agent.hyperparams.get("entropy_coef", 0.01),
            use_top_n_features=agent.use_top_n_features if hasattr(agent, 'use_top_n_features') else self.use_top_n_features,
            curriculum_enabled=agent.curriculum_enabled if hasattr(agent, 'curriculum_enabled') else self.curriculum_enabled,
            opponent_modeling_enabled=agent.opponent_modeling_enabled if hasattr(agent, 'opponent_modeling_enabled') else self.opponent_modeling_enabled
        )
        
        # Create dummy data for initialization
        dummy_state = np.zeros((1, agent.state_dim))
        dummy_action_features = np.zeros((1, agent.action_dim, agent.action_feature_dim))
        
        # Initialize networks by doing a forward pass
        clone.meta_policy(dummy_state)
        for position in agent.positions:
            if position in clone.sub_policies:
                clone.sub_policies[position]([dummy_state, dummy_action_features])
        clone.critic(dummy_state)
        
        # Copy meta policy weights
        clone.meta_policy.set_weights(agent.meta_policy.get_weights())
        
        # Copy sub policy weights - only for positions that exist in both
        for position in agent.positions:
            if position in agent.sub_policies and position in clone.sub_policies:
                clone.sub_policies[position].set_weights(agent.sub_policies[position].get_weights())
        
        # Copy critic weights
        clone.critic.set_weights(agent.critic.get_weights())
        
        # Copy metadata
        clone.hyperparams = copy.deepcopy(agent.hyperparams)
        
        return clone
    
    def _create_fresh_simulator(self, simulator, teams):
        """
        Create a fresh copy of a simulator
        
        Parameters:
        -----------
        simulator : DraftSimulator or SeasonSimulator
            Simulator to copy
            
        Returns:
        --------
        DraftSimulator or SeasonSimulator
            Fresh copy of the simulator
        """
        # For DraftSimulator
        if isinstance(simulator, DraftSimulator):
            # Create a deep copy of players
            fresh_players = copy.deepcopy(simulator.players)
            
            # Reset drafted status
            for player in fresh_players:
                player.is_drafted = False
                player.drafted_round = None
                player.drafted_pick = None
                player.drafted_team = None
            
            # Create a new simulator
            fresh_simulator = DraftSimulator(
                players=fresh_players,
                league_size=simulator.league_size,
                roster_limits=simulator.roster_limits.copy(),
                num_rounds=simulator.num_rounds,
                scoring_settings=simulator.scoring_settings.copy(),
                user_pick=None,
                projection_models=simulator.projection_models
            )
            
            return fresh_simulator
            
        # For SeasonSimulator
        elif isinstance(simulator, SeasonSimulator):
            # Create a new simulator with the same parameters
            fresh_simulator = SeasonSimulator(
                teams=teams,  # Teams will be provided by draft simulator
                num_regular_weeks=simulator.num_regular_weeks,
                num_playoff_teams=simulator.num_playoff_teams,
                num_playoff_weeks=simulator.num_playoff_weeks,
                randomness=simulator.randomness
            )
            
            return fresh_simulator
        
        # Unknown simulator type
        else:
            logger.warning(f"Unknown simulator type: {type(simulator)}")
            return simulator
    
    def calculate_population_diversity(self):
        """
        Calculate diversity metrics for the population
        
        Returns:
        --------
        dict
            Diversity metrics
        """
        # Initialize metrics
        weight_distances = []
        hyperparameter_distances = []
        
        # Calculate pairwise distances between agents
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                agent1 = self.population[i]
                agent2 = self.population[j]
                
                # Only compare agents of the same type
                if type(agent1) == type(agent2):
                    # Calculate weight distance
                    weight_dist = self._calculate_weight_distance(agent1, agent2)
                    weight_distances.append(weight_dist)
                    
                    # Calculate hyperparameter distance
                    hyperparam_dist = self._calculate_hyperparameter_distance(agent1, agent2)
                    hyperparameter_distances.append(hyperparam_dist)
        
        # Calculate average distances
        weight_diversity = np.mean(weight_distances) if weight_distances else 0.0
        hyperparameter_diversity = np.mean(hyperparameter_distances) if hyperparameter_distances else 0.0
        
        # Record diversity metrics
        self.diversity_metrics.append({
            "generation": self.generation,
            "weight_diversity": weight_diversity,
            "hyperparameter_diversity": hyperparameter_diversity
        })
        
        return {
            "weight_diversity": weight_diversity,
            "hyperparameter_diversity": hyperparameter_diversity
        }
    
    def _calculate_weight_distance(self, agent1, agent2):
        """
        Calculate distance between network weights of two agents
        
        Parameters:
        -----------
        agent1 : PPODrafter or HierarchicalPPODrafter
            First agent
        agent2 : PPODrafter or HierarchicalPPODrafter
            Second agent
            
        Returns:
        --------
        float
            Distance between weights
        """
        # For standard PPO agents
        if isinstance(agent1, PPODrafter) and isinstance(agent2, PPODrafter):
            # Calculate distance for actor weights
            actor1_weights = agent1.actor.get_weights()
            actor2_weights = agent2.actor.get_weights()
            
            actor_dist = 0.0
            total_weights = 0
            
            # For each matching layer, calculate L2 distance
            for i in range(min(len(actor1_weights), len(actor2_weights))):
                weights1 = actor1_weights[i]
                weights2 = actor2_weights[i]
                
                if weights1.shape == weights2.shape:
                    # Calculate normalized L2 distance
                    layer_dist = np.mean((weights1 - weights2) ** 2)
                    actor_dist += layer_dist
                    total_weights += 1
            
            # Normalize by number of layers
            if total_weights > 0:
                actor_dist /= total_weights
            
            # Calculate distance for critic weights
            critic1_weights = agent1.critic.get_weights()
            critic2_weights = agent2.critic.get_weights()
            
            critic_dist = 0.0
            total_weights = 0
            
            # For each matching layer, calculate L2 distance
            for i in range(min(len(critic1_weights), len(critic2_weights))):
                weights1 = critic1_weights[i]
                weights2 = critic2_weights[i]
                
                if weights1.shape == weights2.shape:
                    # Calculate normalized L2 distance
                    layer_dist = np.mean((weights1 - weights2) ** 2)
                    critic_dist += layer_dist
                    total_weights += 1
            
            # Normalize by number of layers
            if total_weights > 0:
                critic_dist /= total_weights
            
            # Average actor and critic distances
            return (actor_dist + critic_dist) / 2.0
            
        # For hierarchical PPO agents
        elif isinstance(agent1, HierarchicalPPODrafter) and isinstance(agent2, HierarchicalPPODrafter):
            # Calculate distance for meta policy weights
            meta1_weights = agent1.meta_policy.get_weights()
            meta2_weights = agent2.meta_policy.get_weights()
            
            meta_dist = 0.0
            total_weights = 0
            
            # For each matching layer, calculate L2 distance
            for i in range(min(len(meta1_weights), len(meta2_weights))):
                weights1 = meta1_weights[i]
                weights2 = meta2_weights[i]
                
                if weights1.shape == weights2.shape:
                    # Calculate normalized L2 distance
                    layer_dist = np.mean((weights1 - weights2) ** 2)
                    meta_dist += layer_dist
                    total_weights += 1
            
            # Normalize by number of layers
            if total_weights > 0:
                meta_dist /= total_weights
            
            # Calculate distance for sub policy weights - only matching positions
            sub_dist = 0.0
            total_sub_networks = 0
            
            for position in agent1.positions:
                if position in agent1.sub_policies and position in agent2.sub_policies:
                    sub1_weights = agent1.sub_policies[position].get_weights()
                    sub2_weights = agent2.sub_policies[position].get_weights()
                    
                    pos_dist = 0.0
                    pos_total_weights = 0
                    
                    # For each matching layer, calculate L2 distance
                    for i in range(min(len(sub1_weights), len(sub2_weights))):
                        weights1 = sub1_weights[i]
                        weights2 = sub2_weights[i]
                        
                        if weights1.shape == weights2.shape:
                            # Calculate normalized L2 distance
                            layer_dist = np.mean((weights1 - weights2) ** 2)
                            pos_dist += layer_dist
                            pos_total_weights += 1
                    
                    # Normalize by number of layers
                    if pos_total_weights > 0:
                        pos_dist /= pos_total_weights
                        sub_dist += pos_dist
                        total_sub_networks += 1
            
            # Normalize by number of sub networks
            if total_sub_networks > 0:
                sub_dist /= total_sub_networks
            
            # Calculate distance for critic weights
            critic1_weights = agent1.critic.get_weights()
            critic2_weights = agent2.critic.get_weights()
            
            critic_dist = 0.0
            total_weights = 0
            
            # For each matching layer, calculate L2 distance
            for i in range(min(len(critic1_weights), len(critic2_weights))):
                weights1 = critic1_weights[i]
                weights2 = critic2_weights[i]
                
                if weights1.shape == weights2.shape:
                    # Calculate normalized L2 distance
                    layer_dist = np.mean((weights1 - weights2) ** 2)
                    critic_dist += layer_dist
                    total_weights += 1
            
            # Normalize by number of layers
            if total_weights > 0:
                critic_dist /= total_weights
            
            # Average all distances
            return (meta_dist + sub_dist + critic_dist) / 3.0
        
        # For different agent types
        else:
            # Return max diversity for different agent types
            return 1.0
    
    def _calculate_hyperparameter_distance(self, agent1, agent2):
        """
        Calculate distance between hyperparameters of two agents
        
        Parameters:
        -----------
        agent1 : PPODrafter or HierarchicalPPODrafter
            First agent
        agent2 : PPODrafter or HierarchicalPPODrafter
            Second agent
            
        Returns:
        --------
        float
            Distance between hyperparameters
        """
        # For agents of the same type
        if type(agent1) == type(agent2):
            # Get common hyperparameters
            common_params = []
            for param in agent1.hyperparams:
                if param in agent2.hyperparams:
                    common_params.append(param)
            
            # Calculate normalized relative distance for each parameter
            distances = []
            for param in common_params:
                value1 = agent1.hyperparams[param]
                value2 = agent2.hyperparams[param]
                
                # Avoid division by zero
                if abs(value1) < 1e-10 and abs(value2) < 1e-10:
                    dist = 0.0
                else:
                    # Normalized relative distance
                    avg_value = (abs(value1) + abs(value2)) / 2.0
                    dist = abs(value1 - value2) / (avg_value + 1e-10)
                
                distances.append(dist)
            
            # Return average distance
            return np.mean(distances) if distances else 0.0
        
        # For different agent types
        else:
            # Return max diversity for different agent types
            return 1.0
    
    def visualize_population_metrics(self, output_path=None):
        """
        Visualize population metrics
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the visualization
        """
        plt.figure(figsize=(15, 10))
        
        # Plot fitness over time
        plt.subplot(2, 2, 1)
        
        if self.fitness_history:
            generations = list(range(len(self.fitness_history)))
            
            # Plot individual agent fitness
            for i in range(len(self.population)):
                agent_fitness = [gen_fitness[i] if i < len(gen_fitness) else None for gen_fitness in self.fitness_history]
                valid_points = [(g, f) for g, f in zip(generations, agent_fitness) if f is not None]
                if valid_points:
                    g_vals, f_vals = zip(*valid_points)
                    plt.plot(g_vals, f_vals, 'o-', alpha=0.3, label=f"Agent {i}" if i < 5 else None)
            
            # Plot mean and max fitness
            mean_fitness = [np.mean(gen_fitness) for gen_fitness in self.fitness_history]
            max_fitness = [np.max(gen_fitness) for gen_fitness in self.fitness_history]
            
            plt.plot(generations, mean_fitness, 'k--', linewidth=2, label='Mean Fitness')
            plt.plot(generations, max_fitness, 'r-', linewidth=2, label='Max Fitness')
            
            plt.title('Population Fitness Over Time')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
        
        # Plot diversity metrics
        plt.subplot(2, 2, 2)
        
        if self.diversity_metrics:
            generations = [d["generation"] for d in self.diversity_metrics]
            weight_diversity = [d["weight_diversity"] for d in self.diversity_metrics]
            hyperparam_diversity = [d["hyperparameter_diversity"] for d in self.diversity_metrics]
            
            plt.plot(generations, weight_diversity, 'b-', label='Weight Diversity')
            plt.plot(generations, hyperparam_diversity, 'g-', label='Hyperparameter Diversity')
            
            plt.title('Population Diversity')
            plt.xlabel('Generation')
            plt.ylabel('Diversity')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot agent types
        plt.subplot(2, 2, 3)
        
        # Count agent types
        standard_count = sum(1 for agent in self.population if isinstance(agent, PPODrafter))
        hierarchical_count = sum(1 for agent in self.population if isinstance(agent, HierarchicalPPODrafter))
        
        # Calculate best agent by type
        standard_best = -float('inf')
        hierarchical_best = -float('inf')
        
        for agent in self.population:
            if isinstance(agent, PPODrafter) and agent.fitness > standard_best:
                standard_best = agent.fitness
            elif isinstance(agent, HierarchicalPPODrafter) and agent.fitness > hierarchical_best:
                hierarchical_best = agent.fitness
        
        # Create bar chart
        agent_types = ['Standard', 'Hierarchical']
        counts = [standard_count, hierarchical_count]
        best_fitness = [standard_best, hierarchical_best]
        
        # Plot counts
        bars = plt.bar(agent_types, counts)
        
        # Add best fitness as text
        for i, bar in enumerate(bars):
            if best_fitness[i] > -float('inf'):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'Best: {best_fitness[i]:.2f}', ha='center')
        
        plt.title('Agent Types in Population')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Plot lineage/ancestry of best agent
        plt.subplot(2, 2, 4)
        
        # Placeholder for lineage visualization (will be implemented in save_population)
        plt.text(0.5, 0.5, "Lineage visualization\nwill be saved separately", 
                ha='center', va='center', fontsize=12)
        
        plt.title('Best Agent Lineage')
        plt.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Population metrics visualization saved to {output_path}")
        
        plt.close()
    
    def save_population(self, base_path):
        """
        Save the entire population
        
        Parameters:
        -----------
        base_path : str
            Base path for saving the population
        """
        # Create directory if needed
        os.makedirs(base_path, exist_ok=True)
        
        # Save each agent
        for i, agent in enumerate(self.population):
            # Create agent-specific path
            agent_path = os.path.join(base_path, f"agent_{i}")
            
            # Save agent model
            if isinstance(agent, HierarchicalPPODrafter):
                agent.save_model(agent_path)
            else:
                agent.save_model(agent_path)
            
            # Save agent metadata
            metadata = {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "fitness": agent.fitness,
                "lifetime_episodes": agent.lifetime_episodes,
                "parent_ids": agent.parent_ids,
                "generation": agent.generation,
                "hyperparams": agent.hyperparams,
                "creation_time": agent.creation_time
            }
            
            with open(f"{agent_path}_metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
        
        # Save population metadata
        population_metadata = {
            "generation": self.generation,
            "population_size": self.population_size,
            "best_agent_idx": self.best_agent_idx,
            "best_fitness": self.best_fitness,
            "agents": [agent.agent_id for agent in self.population],
            "timestamp": time.time()
        }
        
        with open(os.path.join(base_path, "population_metadata.json"), "w") as f:
            json.dump(population_metadata, f, indent=4)
        
        # Create and save lineage visualization
        self._visualize_lineage(base_path)
        
        logger.info(f"Population saved to {base_path}")
    
    def _visualize_lineage(self, output_dir):
        """
        Visualize the lineage of agents in the population
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the visualization
        """
        # This is a placeholder - full implementation would require graph visualization
        # libraries like networkx and matplotlib to create a proper lineage graph
        
        # Create a basic text file with lineage information
        with open(os.path.join(output_dir, "lineage.txt"), "w") as f:
            f.write(f"Population Lineage (Generation {self.generation})\n")
            f.write("=" * 50 + "\n\n")
            
            # List all agents
            for i, agent in enumerate(self.population):
                f.write(f"Agent {i}: {agent.agent_id}\n")
                f.write(f"  Type: {agent.agent_type}\n")
                f.write(f"  Fitness: {agent.fitness:.4f}\n")
                f.write(f"  Generation: {agent.generation}\n")
                f.write(f"  Parents: {', '.join(agent.parent_ids) if agent.parent_ids else 'None (initial)'}\n")
                f.write("\n")
    
    def get_best_agent(self):
        """
        Get the best agent from the population
        
        Returns:
        --------
        PPODrafter or HierarchicalPPODrafter
            Best agent
        """
        return self.population[self.best_agent_idx]
    
    def _make_serializable(self, data):
        """
        Convert data to JSON serializable format
        
        Parameters:
        -----------
        data : Any
            Data to convert
            
        Returns:
        --------
        Any
            Serializable data
        """
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items() if k != "best_agent"}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            # For custom objects, return their string representation
            return str(data)