# Handwritten Digits Recognition using Neural Networks

This project uses neural networks to learn to recognize optical handwritten digits

## Getting Started

### Clone the Repository

To get started, clone the repository using Git:

```bash
git clone https://github.com/smebellis/cis581_project2_digits_recognition.git
```

Then, navigate into the project directory:

```bash
cd cis582_project2_digits_recognition
```

### Prerequisites

Ensure you have the following Python packages installed:
- numpy
- pandas
- matplotlib
- scikit-learn

You can install these by running:

```bash
pip install -r requirements.txt
```

*Tip: It is recommended to use a virtual environment (e.g., using `venv` or `conda`) for dependency isolation.*

## How to Run the Code

There are three ways to execute this project:

### 1. Run as a Python Command Line Application

1. **Install Requirements:**  
   Ensure you have installed all the necessary packages (see [Prerequisites](#prerequisites)).
2. **Execute the Script:**  
   Run the Python script from the command line while providing the required arguments. For example:
   ```bash
   python main.py
### 2. Run Using the Provided Bash Script

The project includes a bash script (e.g., `run.sh`) to streamline execution.

1. **Make the Script Executable (on Unix-like systems):**  
   On a new computer, you may need to change the script’s permissions:
   ```bash
   chmod +x run.sh
   ```
2. **Execute the Script:**  
   Run the script from your terminal:
   ```bash
   ./run.sh
   ```
3. **Windows Considerations:**  
   - On Windows, you cannot run bash scripts natively.
   - Install **Git Bash** or use **Windows Subsystem for Linux (WSL)** to run the script.

### 3. Run the PyInstaller Executable

An executable version of the application has been created using PyInstaller. Note the following:

- **Running the Executable:**  
  Simply run the executable file (e.g., `main`) from your command line.
  
- **Windows Compatibility:**  
  Since the executable was compiled using WSL2, it may not run directly on native Windows. In that case, use **WSL2** or a **Linux environment** to execute it.

## Example Execution

#### Using Python Directly:
```bash
python main.py
```

#### Using the Bash Script:
1. On Linux/WSL/Git Bash:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

#### Using the PyInstaller Executable:
```bash
./main
```

*(On Windows, run the executable from a Linux-like environment such as WSL2.)*