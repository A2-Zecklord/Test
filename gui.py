import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox,
    QPushButton, QLabel, QFileDialog, QHBoxLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import subprocess

class TrainingThread(QThread):
    # Signal to indicate completion or error
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        try:
            subprocess.run(self.command, check=True)
            self.finished.emit()
        except subprocess.CalledProcessError as e:
            self.error.emit(str(e))

class TrainingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Training Parameters')
        self.setGeometry(100, 100, 400, 300)

        # Layouts
        main_layout = QVBoxLayout()
        form_layout = QFormLayout()
        
        # Input fields
        self.wav_folder_input = QLineEdit()
        self.wav_folder_input.setPlaceholderText('Path to folder containing WAV files')
        form_layout.addRow("WAV Folder", self.create_browse_button(self.wav_folder_input, folder=True))

        self.csv_file_input = QLineEdit()
        self.csv_file_input.setPlaceholderText('Path to CSV file containing lyrics and prompts')
        form_layout.addRow("CSV File", self.create_browse_button(self.csv_file_input))

        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 512)
        self.batch_size_input.setValue(1)
        form_layout.addRow("Batch Size", self.batch_size_input)

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(100)
        form_layout.addRow("Epochs", self.epochs_input)

        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setRange(0.0000001, 1.0)
        self.learning_rate_input.setDecimals(5)  # Set precision to five decimal places
        self.learning_rate_input.setValue(0.00001)
        self.learning_rate_input.setSingleStep(0.00001)
        form_layout.addRow("Learning Rate", self.learning_rate_input)

        self.log_interval_input = QSpinBox()
        self.log_interval_input.setRange(1, 1000)
        self.log_interval_input.setValue(10)
        form_layout.addRow("Log Interval", self.log_interval_input)

        self.checkpoint_interval_input = QSpinBox()
        self.checkpoint_interval_input.setRange(1, 1000)
        self.checkpoint_interval_input.setValue(5)
        form_layout.addRow("Checkpoint Interval", self.checkpoint_interval_input)

        self.checkpoint_dir_input = QLineEdit()
        self.checkpoint_dir_input.setPlaceholderText('Directory to save checkpoints')
        form_layout.addRow("Checkpoint Directory", self.create_browse_button(self.checkpoint_dir_input, folder=True))

        # Run button
        self.run_button = QPushButton('Start Training')
        self.run_button.clicked.connect(self.start_training)

        # Status label
        self.status_label = QLabel('')
        self.status_label.setAlignment(Qt.AlignCenter)

        # Add form layout and button to main layout
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.run_button)
        main_layout.addWidget(self.status_label)
        
        # Set main layout
        self.setLayout(main_layout)

    def create_browse_button(self, input_field, folder=False):
        layout = QHBoxLayout()
        layout.addWidget(input_field)
        browse_button = QPushButton('Browse')
        browse_button.clicked.connect(lambda: self.browse(input_field, folder))
        layout.addWidget(browse_button)
        return layout

    def browse(self, input_field, folder=False):
        file_dialog = QFileDialog()
        if folder:
            folder_path = file_dialog.getExistingDirectory(self, "Select Folder")
            if folder_path:
                input_field.setText(folder_path)
        else:
            file_path, _ = file_dialog.getOpenFileName(self, "Select File", "", "All Files (*)")
            if file_path:
                input_field.setText(file_path)

    def start_training(self):
        # Collect arguments
        wav_folder = self.wav_folder_input.text()
        csv_file = self.csv_file_input.text()
        batch_size = self.batch_size_input.value()
        epochs = self.epochs_input.value()
        learning_rate = self.learning_rate_input.value()
        log_interval = self.log_interval_input.value()
        checkpoint_interval = self.checkpoint_interval_input.value()
        checkpoint_dir = self.checkpoint_dir_input.text()

        # Command to run training script
        command = [
            'python', 'training.py',
            '--wav_folder', wav_folder,
            '--csv_file', csv_file,
            '--batch_size', str(batch_size),
            '--epochs', str(epochs),
            '--learning_rate', str(learning_rate),
            '--log_interval', str(log_interval),
            '--checkpoint_interval', str(checkpoint_interval),
            '--checkpoint_dir', checkpoint_dir,
        ]
        
        # Disable the Run button and update status
        self.run_button.setEnabled(False)
        self.status_label.setText("Training in progress...")

        # Create and start the training thread
        self.training_thread = TrainingThread(command)
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.error.connect(self.on_training_error)
        self.training_thread.start()

    def on_training_finished(self):
        # Update status and re-enable the Run button
        self.status_label.setText("Training completed successfully.")
        self.run_button.setEnabled(True)

    def on_training_error(self, error_msg):
        # Display error message and re-enable the Run button
        self.status_label.setText(f"Error during training: {error_msg}")
        self.run_button.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrainingApp()
    window.show()
    sys.exit(app.exec_())
