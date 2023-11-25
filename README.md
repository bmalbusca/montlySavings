# Monthly Savings

## Overview

The **Monthly Savings** program is a Python script designed to process and parse bank extracts, providing a user-friendly data visualization interface. It aims to help users analyze and manage their monthly savings by presenting the financial data in a clear and comprehensible manner.

## Current Features

- **Bank Extract Parsing:** Parses PDF extracts and CSV files from Santander totta bank into Pandas Dataframe
   - Filters: largest debt receivers; largest expenses; largest recurring expenses and receivers
   - Results: Period Income versus Expenses
  
- **Data Visualization Interface:** Generates visual representations of monthly savings data for easy analysis trough Matplotlib.
   - Plot: Accumulated earnings; Earnings; Expenses; Accumulated Expenses; Accumulated difference; Savings result


## How to Use

### Prerequisites

- Python 3.x installed on your machine.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/bmalbusca/monthlySavings.git
   ```

2. Navigate to the project directory:

   ```bash
   cd monthlySavings
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Development Usage

1. Run the script:

   ```bash
   python monthlySavings.py
   ```

## Dependencies

- matplotlib: For creating visualizations.
- pandas: For data manipulation and analysis.
- tabula: For PDF parsing

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the open-source community for providing invaluable tools and resources.

---

Feel free to customize this README to better suit your project's specifics. Add more detailed instructions, examples, or any other information that would be helpful for users and contributors.