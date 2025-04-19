# src/gui/draft_gui_pyqt.py

import sys
import logging
import os
import pandas as pd
import numpy as np
import pickle

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox, QSplitter,
        QFrame, QSizePolicy, QTableWidget, QTableWidgetItem, QAbstractItemView,
        QHeaderView, QMenuBar, QFileDialog, QComboBox, QRadioButton, QButtonGroup
    )
    from PyQt6.QtCore import Qt, pyqtSlot
    from PyQt6.QtGui import QFont, QTextCursor, QColor, QAction
    PYQT_AVAILABLE = True
except ImportError:
    print("ERROR: PyQt6 is not installed. Please run 'pip install PyQt6'")
    PYQT_AVAILABLE = False

from .draft_manager import DraftManager

class NumericTableWidgetItem(QTableWidgetItem):
    """Custom QTableWidgetItem that allows for proper numeric sorting."""
    def __init__(self, text, numeric_value):
        super().__init__(text)
        self.numeric_value = numeric_value

    def __lt__(self, other):
        # Override less-than comparison for sorting
        if isinstance(other, NumericTableWidgetItem):
            return self.numeric_value < other.numeric_value
        # Fallback for comparing with non-numeric items if necessary
        try:
            # Attempt numeric comparison if possible, otherwise use text
            return self.numeric_value < float(other.text())
        except (ValueError, TypeError):
            return super().__lt__(other)

# --- GUI Class ---
class DraftApp(QMainWindow):
    def __init__(self, config):
        super().__init__()
        if not PYQT_AVAILABLE:
            raise ImportError("PyQt6 library is required but not installed.")

        self.config = config
        self.logger = logging.getLogger("fantasy_football")
        self.current_recommendations = []
        self.draft_log = []

        try:
            models_dir = config['paths']['models_dir']
            model_filename = config['rl_training'].get('model_save_path', "ppo_fantasy_draft_agent") + ".zip"
            model_path = os.path.join(models_dir, model_filename)
            settings_path = config['paths']['config_path']
            projections_dir = models_dir
            agent_pos = config['rl_training'].get('agent_draft_position', 1)

            self.manager = DraftManager(
                projections_path=projections_dir,
                league_settings_path=settings_path,
                model_path=model_path,
                agent_draft_pos=agent_pos
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Draft Manager: {e}", exc_info=True)
            if not QApplication.instance(): QApplication(sys.argv)
            self.show_error("Initialization Error", f"Could not load draft manager:\n{e}")
            sys.exit(1)

        self.initUI()
        self._update_display()

    def initUI(self):
        self.setWindowTitle("Fantasy Football Draft Assistant (PyQt)")
        self.setGeometry(100, 100, 1400, 900)

        self.create_menu_bar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Status Bar
        self.status_label = QLabel("Loading...")
        font = self.status_label.font(); font.setPointSize(11); font.setBold(True)
        self.status_label.setFont(font)
        self.status_label.setStyleSheet("background-color: #e1e1e1; padding: 5px; border: 1px solid #c0c0c0; border-radius: 3px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        # Control Frame
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)
        control_layout.setContentsMargins(0,0,0,0)
        control_layout.addWidget(QLabel("Enter Pick:"))
        self.pick_entry = QLineEdit()
        self.pick_entry.setPlaceholderText("Enter player name OR double-click list...")
        control_layout.addWidget(self.pick_entry, 1)
        self.submit_button = QPushButton("Submit Pick")
        control_layout.addWidget(self.submit_button)
        main_layout.addWidget(control_frame)

        # Display Area Splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Pane
        left_pane = QFrame()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0,0,0,0)

        filter_frame = QFrame()
        filter_layout = QHBoxLayout(filter_frame)
        filter_layout.addWidget(QLabel("Filter:"))
        self.pos_filter_group = QButtonGroup(self)
        positions = ["ALL", "QB", "RB", "WR", "TE", "FLEX"]
        for pos in positions:
            radio_button = QRadioButton(pos)
            filter_layout.addWidget(radio_button)
            self.pos_filter_group.addButton(radio_button)
            if pos == "ALL": radio_button.setChecked(True)
        self.pos_filter_group.buttonClicked.connect(self._filter_available_players)
        filter_layout.addStretch()
        left_layout.addWidget(filter_frame)

        left_layout.addWidget(QLabel("Available Players:"))
        self.available_players_table = QTableWidget()
        self._configure_table(self.available_players_table)
        self.available_players_table.itemDoubleClicked.connect(self._available_double_clicked)
        self.available_players_table.itemSelectionChanged.connect(self._display_player_details)
        left_layout.addWidget(self.available_players_table, 1)

        # Middle Pane
        middle_pane = QFrame()
        middle_layout = QVBoxLayout(middle_pane)
        middle_layout.setContentsMargins(0,0,0,0)

        middle_layout.addWidget(QLabel("Agent Recommendations:"))
        self.recommendations_table = QTableWidget()
        self._configure_table(self.recommendations_table)
        self.recommendations_table.itemDoubleClicked.connect(self._recommendation_double_clicked)
        self.recommendations_table.itemSelectionChanged.connect(self._display_player_details)
        middle_layout.addWidget(self.recommendations_table, 1)

        middle_layout.addWidget(QLabel("Selected Player Details:"))
        self.player_details_text = QTextEdit()
        self.player_details_text.setReadOnly(True)
        self.player_details_text.setFont(QFont("Consolas", 9))
        self.player_details_text.setFixedHeight(120) # Increased height
        middle_layout.addWidget(self.player_details_text)

        middle_layout.addWidget(QLabel("Next Opponent Watchlist:"))
        self.opponent_watchlist_text = QTextEdit()
        self.opponent_watchlist_text.setReadOnly(True)
        self.opponent_watchlist_text.setFont(QFont("Consolas", 9))
        self.opponent_watchlist_text.setFixedHeight(120)
        middle_layout.addWidget(self.opponent_watchlist_text)

        # Right Pane
        right_pane = QFrame()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0,0,0,0)

        right_layout.addWidget(QLabel("Your Roster:"))
        self.agent_roster_text = QTextEdit()
        self.agent_roster_text.setReadOnly(True)
        self.agent_roster_text.setFont(QFont("Consolas", 9))
        right_layout.addWidget(self.agent_roster_text, 1)

        opponent_frame = QFrame()
        opponent_layout = QHBoxLayout(opponent_frame)
        opponent_layout.addWidget(QLabel("View Team:"))
        self.opponent_select_combo = QComboBox()
        opponent_layout.addWidget(self.opponent_select_combo, 1)
        right_layout.addWidget(opponent_frame)

        self.opponent_roster_text = QTextEdit()
        self.opponent_roster_text.setReadOnly(True)
        self.opponent_roster_text.setFont(QFont("Consolas", 9))
        right_layout.addWidget(self.opponent_roster_text, 1)

        right_layout.addWidget(QLabel("Draft Log:"))
        self.draft_log_text = QTextEdit()
        self.draft_log_text.setReadOnly(True)
        self.draft_log_text.setFont(QFont("Consolas", 9))
        right_layout.addWidget(self.draft_log_text, 1)

        main_splitter.addWidget(left_pane)
        main_splitter.addWidget(middle_pane)
        main_splitter.addWidget(right_pane)
        main_splitter.setStretchFactor(0, 4) # Available players largest
        main_splitter.setStretchFactor(1, 3) # Middle pane
        main_splitter.setStretchFactor(2, 2) # Rosters/Log
        main_layout.addWidget(main_splitter, 1)

        self.connect_signals()
        self._populate_opponent_selector()

    def _configure_table(self, table_widget):
        """Applies common configurations to QTableWidgets, including sorting."""
        table_widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table_widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table_widget.verticalHeader().setVisible(False)
        # table_widget.horizontalHeader().setStretchLastSection(True) # Disable for better auto-sizing
        table_widget.setAlternatingRowColors(True)
        table_widget.setShowGrid(True)
        table_widget.setSortingEnabled(True) # <<< Enable Sorting

    def create_menu_bar(self):
        """Creates the main menu bar."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')

        save_action = QAction('&Save Draft State', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self._save_draft)
        file_menu.addAction(save_action)

        load_action = QAction('&Load Draft State', self)
        load_action.setShortcut('Ctrl+L')
        load_action.triggered.connect(self._load_draft)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        exit_action = QAction('&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def connect_signals(self):
        """Connect widget signals to slots."""
        self.submit_button.clicked.connect(self._handle_submit_pick)
        self.pick_entry.returnPressed.connect(self.submit_button.click)
        self.opponent_select_combo.currentIndexChanged.connect(self._display_opponent_roster)
        # Double-click signals connected in initUI
        # Filter signal connected in initUI

    # --- Double-Click Handlers ---
    @pyqtSlot(QTableWidgetItem)
    def _recommendation_double_clicked(self, item):
        """Handle double-click on recommendation table."""
        if not self.manager.is_agent_turn(): return
        row = item.row()
        if self.recommendations_table.columnCount() > 0:
            name_item = self.recommendations_table.item(row, 0)
            if name_item:
                self.pick_entry.setText(name_item.text())
                self._handle_submit_pick()
        else:
            self.logger.warning("Recommendation table has no columns to get name from.")

    @pyqtSlot(QTableWidgetItem)
    def _available_double_clicked(self, item):
        """Handle double-click on available players table."""
        row = item.row()
        if self.available_players_table.columnCount() > 0:
            name_item = self.available_players_table.item(row, 0)
            if name_item:
                self.pick_entry.setText(name_item.text())
        else:
            self.logger.warning("Available players table has no columns to get name from.")

    # --- Update and Display Logic ---
    def _update_display(self):
        """Updates all GUI elements based on the current draft state."""
        if self.manager.is_draft_complete():
            self.status_label.setText("Draft Complete!")
            self.pick_entry.setEnabled(False)
            self.submit_button.setEnabled(False)
            self.recommendations_table.setRowCount(0)
            self.opponent_watchlist_text.clear()
            # Disable filters
            for button in self.pos_filter_group.buttons():
                button.setEnabled(False)
            return

        round_num, pick_in_round, current_team_id = self.manager.get_current_status()
        is_agent = self.manager.is_agent_turn()
        turn_text = f"(Your Turn - Team {self.manager.agent_team_id})" if is_agent else f"(Team {current_team_id}'s Turn)"
        self.status_label.setText(f"Round: {round_num} | Pick: {pick_in_round} / {self.manager.num_teams} {turn_text}")

        selected_button = self.pos_filter_group.checkedButton()
        pos_filter = selected_button.text() if selected_button else "ALL"

        # Get recommendations first to know who to highlight
        self.current_recommendations = self.manager.get_agent_recommendations(top_n=15) if is_agent else []
        recommended_ids = {p.get('player_id', None) for p in self.current_recommendations} # Use get for safety

        # Update Available Players Table
        available_df_full = self.manager.get_available_players(limit=250) # Get more for filtering

        if pos_filter == "ALL": available_df_filtered = available_df_full
        elif pos_filter == "FLEX":
            flex_positions = ['RB', 'WR', 'TE']
            if self.manager.has_op_slot: flex_positions.append('QB')
            available_df_filtered = available_df_full[available_df_full['position'].isin(flex_positions)]
        else:
            available_df_filtered = available_df_full[available_df_full['position'] == pos_filter]

        # Define columns for available table, INCLUDING NEW ONES
        available_cols = ['name', 'position', 'risk_adjusted_vorp', 'vorp', 'projected_points', 'projection_low', 'projection_high', 'ceiling_projection', 'age']
        self._populate_table(self.available_players_table, available_df_filtered.head(100), available_cols, highlight_ids=recommended_ids)

        # Update Agent Roster List
        agent_roster = self.manager.get_agent_roster()
        self._display_roster_text(self.agent_roster_text, pd.DataFrame(agent_roster), ['name', 'position', 'vorp', 'risk_adjusted_vorp'])

        # Update Agent Recommendations & Controls
        if is_agent:
            self.pick_entry.setPlaceholderText("Enter YOUR player OR double-click list...")
            self.submit_button.setText("Select My Pick")
            self.submit_button.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
            self.pick_entry.setEnabled(True); self.submit_button.setEnabled(True)
            self.pick_entry.setFocus()
            # Define columns for recommendation table
            rec_cols = ['name', 'position', 'risk_adjusted_vorp', 'vorp', 'projected_points', 'projection_low', 'projection_high', 'ceiling_projection']
            self._populate_table(self.recommendations_table, pd.DataFrame(self.current_recommendations), rec_cols)
            self.logger.debug(f"Displayed {len(self.current_recommendations)} recommendations.")
            self.opponent_watchlist_text.clear()
        else:
            self.pick_entry.setPlaceholderText("Enter OPPONENT'S player OR double-click list...")
            self.submit_button.setText("Submit Opponent Pick")
            self.submit_button.setStyleSheet("")
            self.pick_entry.setEnabled(True); self.submit_button.setEnabled(True)
            self.pick_entry.setFocus()
            self.recommendations_table.setRowCount(0)
            self._update_opponent_watchlist()

        # Enable/disable filters based on turn
        for button in self.pos_filter_group.buttons():
            button.setEnabled(True) # Always enable filters for browsing

        self._display_opponent_roster()
        self._display_player_details() # Update details possibly based on filter/selection change

    def _populate_table(self, table_widget, df, cols, highlight_ids=None):
        """Populates a QTableWidget, handles numeric sorting, and text highlighting."""
        highlight_ids = highlight_ids or set()
        highlight_color = QColor('darkGreen') # Color for recommended player text

        # Disable sorting before clearing/populating to prevent issues
        table_widget.setSortingEnabled(False)

        if df is None or df.empty:
            table_widget.setRowCount(0)
            table_widget.setColumnCount(len(cols))
            table_widget.setHorizontalHeaderLabels([c.replace('_',' ').title() for c in cols])
            table_widget.setSortingEnabled(True) # Re-enable sorting
            return

        display_cols = [col for col in cols if col in df.columns]
        if not display_cols:
            table_widget.setRowCount(0); table_widget.setColumnCount(0)
            table_widget.setSortingEnabled(True) # Re-enable sorting
            return

        # Include player_id for highlighting logic, even if not displayed
        fetch_cols = display_cols[:]
        if 'player_id' not in fetch_cols: fetch_cols.append('player_id')
        fetch_cols = [col for col in fetch_cols if col in df.columns] # Ensure all are valid
        df_display = df[fetch_cols].copy()

        table_widget.setRowCount(len(df_display))
        table_widget.setColumnCount(len(display_cols)) # Only display requested cols
        table_widget.setHorizontalHeaderLabels([c.replace('_',' ').title() for c in display_cols])

        for row_idx, row_data in enumerate(df_display.itertuples(index=False)):
            player_id = getattr(row_data, 'player_id', None)
            should_highlight = player_id in highlight_ids

            for col_idx, col_name in enumerate(display_cols):
                cell_data = getattr(row_data, col_name, None)
                item_text = ""
                item = None

                # Handle different data types for display and sorting item creation
                if isinstance(cell_data, (int, float, np.number)):
                    numeric_val = float(cell_data) if pd.notna(cell_data) else 0.0 # Store original numeric value
                    item_text = f"{numeric_val:.2f}" if pd.notna(cell_data) else "N/A"
                    # Use custom item for numeric sorting
                    item = NumericTableWidgetItem(item_text, numeric_val)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item_text = str(cell_data) if pd.notna(cell_data) else "N/A"
                    # Use standard item for text sorting
                    item = QTableWidgetItem(item_text)

                # Apply text highlighting
                if should_highlight:
                    item.setForeground(highlight_color) # Set text color

                table_widget.setItem(row_idx, col_idx, item)

        # Adjust column widths after populating
        table_widget.resizeColumnsToContents()
        try:
            name_col_index = display_cols.index('name')
            table_widget.setColumnWidth(name_col_index, 180)
        except ValueError: pass
        # table_widget.horizontalHeader().setStretchLastSection(True) # Often better without this

        # Re-enable sorting after population
        table_widget.setSortingEnabled(True)

    # --- Other methods (_display_roster_text, _clear_text_widget, _handle_submit_pick, _populate_opponent_selector, _display_opponent_roster, _display_player_details, _add_to_draft_log, _update_draft_log_display, _update_opponent_watchlist, show_error, show_warning, ask_confirmation, _save_draft, _load_draft, closeEvent) remain largely the same as in the previous version ---
    # Make sure _display_players_text is kept if used by _update_opponent_watchlist
    def _display_players_text(self, text_widget, players_df, cols, title=None):
        """Formats and displays player data in a QTextEdit widget (like roster)."""
        text_widget.clear()
        if title:
            text_widget.setFontWeight(QFont.Weight.Bold)
            text_widget.append(title)
            text_widget.setFontWeight(QFont.Weight.Normal)
            text_widget.append("-" * len(title))

        if players_df is None or players_df.empty:
            text_widget.append("No players to display.")
            return

        display_cols = [col for col in cols if col in players_df.columns]
        if not display_cols:
            text_widget.append("No relevant player columns found.")
            return

        formatted_df = players_df[display_cols].copy()
        # Adjusted widths for watchlist context
        col_widths = {'name': 25, 'position': 6, 'adjusted_score': 10, 'combined_score': 8}
        row_fmts = []
        header_titles = []

        for col in display_cols:
            width = col_widths.get(col, 12)
            row_fmts.append("{:<" + str(width) + "}")
            header_titles.append(col.replace('_',' ').title())
            if formatted_df[col].dtype == 'float64' or formatted_df[col].dtype == 'float32':
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            else:
                formatted_df[col] = formatted_df[col].astype(str)

        header_fmt_string = " ".join(row_fmts)
        row_fmt_string = " ".join(row_fmts)

        try:
            header_line = header_fmt_string.format(*header_titles)
            display_string = header_line + "\n" + "-" * len(header_line) + "\n"
            for _, row in formatted_df.iterrows():
                row_values = [row[col] for col in display_cols]
                display_string += row_fmt_string.format(*row_values) + "\n"
            text_widget.setText(display_string)
            text_widget.moveCursor(QTextCursor.MoveOperation.Start)
        except Exception as fmt_err:
            self.logger.error(f"Error formatting players text display: {fmt_err}", exc_info=True)
            text_widget.append("Error displaying player data.")

    def _display_roster_text(self, text_widget, players_df, cols):
        """Formats and displays the agent roster in a QTextEdit widget."""
        text_widget.clear()
        if players_df is None or players_df.empty:
            text_widget.append("Roster is empty.")
            return

        display_cols = [col for col in cols if col in players_df.columns]
        if not display_cols:
            text_widget.append("No relevant roster columns found.")
            return

        formatted_df = players_df[display_cols].copy()
        col_widths = {'name': 25, 'position': 6, 'risk_adjusted_vorp': 10, 'vorp': 8} # Adjusted
        row_fmts = []
        header_titles = []

        for col in display_cols:
            width = col_widths.get(col, 12)
            row_fmts.append("{:<" + str(width) + "}")
            header_titles.append(col.replace('_',' ').title())
            if formatted_df[col].dtype == 'float64' or formatted_df[col].dtype == 'float32':
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            else:
                formatted_df[col] = formatted_df[col].astype(str)

        header_fmt_string = " ".join(row_fmts)
        row_fmt_string = " ".join(row_fmts)

        try:
            header_line = header_fmt_string.format(*header_titles)
            display_string = header_line + "\n" + "-" * len(header_line) + "\n"
            for _, row in formatted_df.iterrows():
                row_values = [row[col] for col in display_cols]
                display_string += row_fmt_string.format(*row_values) + "\n"
            text_widget.setText(display_string)
            text_widget.moveCursor(QTextCursor.MoveOperation.Start)
        except Exception as fmt_err:
            self.logger.error(f"Error formatting roster display: {fmt_err}", exc_info=True)
            text_widget.append("Error displaying roster data.")

    def _clear_text_widget(self, text_widget):
        """Clears a QTextEdit widget."""
        text_widget.clear()

    @pyqtSlot()
    def _handle_submit_pick(self):
        """Handles the submission of either an opponent or agent pick."""
        if self.manager.is_draft_complete(): return

        player_query = self.pick_entry.text().strip()
        if not player_query:
            self.show_warning("Input Error", "Please enter the player name.")
            return

        player_dict = self.manager.find_player(player_query)

        if not player_dict:
            self.show_error("Player Not Found", f"Could not find available player matching '{player_query}'. Check spelling or availability.")
            return

        if player_dict.get('player_id') in self.manager.drafted_player_ids:
            self.show_error("Player Already Drafted", f"'{player_dict['name']}' has already been drafted.")
            return

        pick_num_before_action = self.manager.current_pick_overall

        if self.manager.is_agent_turn():
            if self.ask_confirmation("Confirm Your Pick", f"Confirm your pick:\n{player_dict['name']} ({player_dict['position']})?"):
                if self.manager.make_agent_pick(player_dict):
                    self._add_to_draft_log(pick_num_before_action, self.manager.agent_team_id, player_dict.get('name','?'), player_dict.get('position','?'))
                    self.pick_entry.clear()
                    self._update_display()
                else:
                    self.show_error("Pick Error", "Failed to record your pick.")
        else:
            current_team_id = self.manager.get_current_status()[2]
            if self.ask_confirmation("Confirm Opponent Pick", f"Confirm Team {current_team_id} pick:\n{player_dict['name']} ({player_dict['position']})?"):
                if self.manager.make_opponent_pick(player_dict):
                    self._add_to_draft_log(pick_num_before_action, current_team_id, player_dict.get('name','?'), player_dict.get('position','?'))
                    self.pick_entry.clear()
                    self._update_display()
                else:
                    self.show_error("Pick Error", "Failed to record opponent pick.")

    @pyqtSlot()
    def _filter_available_players(self):
        """Filters the available players table based on radio button selection."""
        self._update_display()

    @pyqtSlot()
    def _display_player_details(self):
        """Displays details for the selected player in either table."""
        player_name = None
        source_table = None

        # Determine which table has the selection
        if self.available_players_table.hasFocus() or len(self.available_players_table.selectedItems()) > 0:
            selected_items = self.available_players_table.selectedItems()
            if selected_items:
                source_table = self.available_players_table
                name_item = source_table.item(selected_items[0].row(), 0) # Name in col 0
                if name_item: player_name = name_item.text()
        elif self.recommendations_table.hasFocus() or len(self.recommendations_table.selectedItems()) > 0:
            selected_items = self.recommendations_table.selectedItems()
            if selected_items:
                source_table = self.recommendations_table
                name_item = source_table.item(selected_items[0].row(), 0) # Name in col 0
                if name_item: player_name = name_item.text()

        if player_name:
            player_details_row = self.manager.projections_master[
                self.manager.projections_master['name'] == player_name
            ]
            if not player_details_row.empty:
                details = player_details_row.iloc[0].to_dict()
                detail_str = f"Name: {details.get('name', 'N/A')}\n" \
                            f"Pos: {details.get('position', 'N/A')} | Age: {details.get('age', 'N/A')}\n" \
                            f"Proj Pts: {details.get('projected_points', 0.0):.2f}\n" \
                            f"VORP: {details.get('vorp', 0.0):.2f} | RiskAdj: {details.get('risk_adjusted_vorp', 0.0):.2f}\n" \
                            f"Range: {details.get('projection_low', 0.0):.2f} - {details.get('projection_high', 0.0):.2f}\n" \
                            f"Ceiling: {details.get('ceiling_projection', 0.0):.2f}"
                self.player_details_text.setText(detail_str)
                return

        self.player_details_text.clear() # Clear if no valid selection or details found

    @pyqtSlot()
    def _display_opponent_roster(self):
        """Displays the roster for the selected opponent."""
        selected_team_id = self.opponent_select_combo.currentData()
        if selected_team_id is None or selected_team_id == -1:
            self.opponent_roster_text.clear()
            self.opponent_roster_text.append("Select an opponent team above.")
            return

        roster = self.manager.get_team_roster(selected_team_id)
        self._display_roster_text(self.opponent_roster_text, pd.DataFrame(roster), ['name', 'position', 'vorp'])

    def _add_to_draft_log(self, pick_num, team_id, player_name, player_pos):
        """Adds an entry to the internal draft log."""
        round_num = (pick_num // self.manager.num_teams) + 1
        pick_in_round = (pick_num % self.manager.num_teams) + 1
        log_entry = f"R{round_num:02d}.{pick_in_round:02d} (Pk {pick_num+1}) Team {team_id}: {player_name} ({player_pos})"
        self.draft_log.append(log_entry)
        self._update_draft_log_display()

    def _update_draft_log_display(self):
        """Updates the draft log text widget."""
        self.draft_log_text.clear()
        self.draft_log_text.setText("\n".join(self.draft_log))
        self.draft_log_text.moveCursor(QTextCursor.MoveOperation.End)

    def _update_opponent_watchlist(self):
        """Updates the watchlist for the *next* opponent."""
        if self.manager.is_draft_complete() or self.manager.is_agent_turn():
            self.opponent_watchlist_text.clear()
            return

        next_pick_index = self.manager.current_pick_overall
        if next_pick_index >= len(self.manager.draft_order): return # Avoid index error if draft ended mid-update

        next_team_id = self.manager.draft_order[next_pick_index]

        watchlist = self.manager.predict_opponent_targets(next_team_id, num_targets=5)
        title = f"--- Team {next_team_id} Watchlist (Predicted) ---"
        # Display using the text helper
        self._display_players_text(self.opponent_watchlist_text, pd.DataFrame(watchlist),
                                    ['name', 'position', 'adjusted_score', 'combined_score'], title=title)


    @pyqtSlot()
    def _save_draft(self):
        """Saves the current draft state to a file."""
        if self.manager.is_draft_complete():
            self.show_warning("Save Draft", "Draft is already complete.")
            return
        default_filename = f"draft_state_league_{self.manager.league_settings.get('league_id', 'unknown')}_pick_{self.manager.current_pick_overall}.pkl"
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Draft State", default_filename, "Pickle Files (*.pkl)")
        if filepath:
            try:
                snapshot = self.manager.get_state_snapshot()
                with open(filepath, 'wb') as f: pickle.dump(snapshot, f)
                self.logger.info(f"Draft state saved successfully to: {filepath}")
                QMessageBox.information(self, "Save Successful", f"Draft state saved to:\n{filepath}")
            except Exception as e:
                self.logger.error(f"Failed to save draft state: {e}", exc_info=True)
                self.show_error("Save Failed", f"Could not save draft state:\n{e}")

    @pyqtSlot()
    def _load_draft(self):
        """Loads a draft state from a file."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Draft State", "", "Pickle Files (*.pkl)")
        if filepath:
            confirm = self.ask_confirmation("Load Draft", "Loading state will overwrite current progress. Continue?")
            if not confirm: return
            try:
                with open(filepath, 'rb') as f: snapshot = pickle.load(f)
                if self.manager.load_state_snapshot(snapshot):
                    self.logger.info(f"Draft state loaded successfully from: {filepath}")
                    # Rebuild draft log from loaded state
                    self.draft_log = []
                    pick_num = 0
                    while pick_num < self.manager.current_pick_overall:
                        team_id = self.manager.draft_order[pick_num]
                        # Find the player picked at this pick number for this team
                        # This assumes rosters store players in draft order for that team
                        roster = self.manager.teams_rosters.get(team_id, [])
                        team_pick_index = sum(1 for i in range(pick_num + 1) if self.manager.draft_order[i] == team_id) - 1
                        if team_pick_index < len(roster):
                            player = roster[team_pick_index]
                            self._add_to_draft_log(pick_num, team_id, player.get('name','?'), player.get('position','?'))
                        else:
                            self.logger.warning(f"Could not reconstruct log for pick {pick_num+1}, team {team_id}")
                        pick_num += 1

                    self._update_draft_log_display()
                    self._update_display()
                    QMessageBox.information(self, "Load Successful", f"Draft state loaded from:\n{filepath}")
                else:
                    self.show_error("Load Failed", "Failed to apply the loaded draft state.")
            except Exception as e:
                self.logger.error(f"Failed to load draft state: {e}", exc_info=True)
                self.show_error("Load Failed", f"Could not load draft state file:\n{e}")

    def show_error(self, title, message): QMessageBox.critical(self, title, message)
    def show_warning(self, title, message): QMessageBox.warning(self, title, message)
    def ask_confirmation(self, title, message):
        reply = QMessageBox.question(self, title, message, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        return reply == QMessageBox.StandardButton.Yes
    def closeEvent(self, event): self.logger.info("Draft GUI closing."); event.accept()
    
    def _populate_opponent_selector(self):
        """Populates the opponent team selection dropdown."""
        current_selection_data = self.opponent_select_combo.currentData() # Store current selection

        self.opponent_select_combo.blockSignals(True) # Prevent signal emission during repopulation
        self.opponent_select_combo.clear()
        # Add "Select Team" prompt
        self.opponent_select_combo.addItem("--- Select Team ---", -1) # Use -1 or None as data for placeholder
        # Get team info (assuming team_id and team_name)
        teams = self.manager.league_settings.get('teams', [])
        found_current = False
        current_index = 0 # Default to placeholder
        if teams: # Check if teams list is not empty
            sorted_teams = sorted(teams, key=lambda t: t.get('team_id', 0))
            for i, team_info in enumerate(sorted_teams):
                team_id = team_info.get('team_id')
                team_name = team_info.get('team_name', f"Team {team_id}")
                if team_id and team_id != self.manager.agent_team_id: # Exclude agent
                    self.opponent_select_combo.addItem(f"{team_name} (ID: {team_id})", team_id)
                    # Check if this was the previously selected item
                    if team_id == current_selection_data:
                        current_index = self.opponent_select_combo.count() - 1 # Get index of the added item
                        found_current = True
        else:
            self.logger.warning("No teams found in league settings to populate opponent selector.")

        self.opponent_select_combo.blockSignals(False) # Re-enable signals

        # Restore previous selection if possible
        if found_current:
            self.opponent_select_combo.setCurrentIndex(current_index)
        else:
            self.opponent_select_combo.setCurrentIndex(0) # Default to placeholder
            # Explicitly clear opponent roster if selection reset to placeholder
            if current_selection_data != -1 and current_selection_data is not None:
                self._display_opponent_roster() # Trigger display update which should clear the roster text


def run_gui(config):
    """Initializes and runs the PyQt application."""
    if not PYQT_AVAILABLE:
        logging.getLogger("fantasy_football").critical("PyQt6 not found. Cannot run GUI.")
        return 1
    app = QApplication(sys.argv)
    try:
        ex = DraftApp(config)
        ex.show()
        return app.exec()
    except Exception as e:
        logging.getLogger("fantasy_football").critical(f"Error launching Draft GUI: {e}", exc_info=True)
        if QApplication.instance():
            msg_box = QMessageBox(); msg_box.setIcon(QMessageBox.Icon.Critical); msg_box.setText("Fatal Error")
            msg_box.setInformativeText(f"Could not launch the Draft Assistant:\n{e}"); msg_box.setWindowTitle("Launch Error"); msg_box.exec()
        else: print(f"FATAL ERROR launching GUI: {e}")
        return 1


if __name__ == '__main__':
    print("Starting Draft GUI (PyQt) directly...")
    test_config = {
        'paths': { 'models_dir': 'data/models', 'config_path': 'configs/league_settings.json', 'output_dir': 'data/outputs' },
        'rl_training': { 'model_save_path': 'ppo_fantasy_draft_agent', 'agent_draft_position': 5 },
        'logging': { 'level': "INFO", 'file': "draft_gui_pyqt_run.log" }
    }
    logging.basicConfig( level=test_config['logging'].get('level', 'INFO').upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[ logging.StreamHandler(sys.stdout), logging.FileHandler(test_config['logging'].get('file', 'gui_direct.log'), mode='a') ] )
    logger_main = logging.getLogger("fantasy_football")
    logger_main.info("--- Starting Draft GUI (PyQt - Direct Run) ---")
    exit_code = run_gui(test_config)
    sys.exit(exit_code)