# MLDS Hackathon Team: Messi Data

Team Members: John He, Daeun Ji, Teddy Debreu, and Kavya Bhat

Hello, thank you for checking out our repository!

## Messi Data’s Playbook: Optimizing Lineups and Substitutions with AI
This project focuses on analyzing soccer match data and developing an AI-powered lineup recommendation and substitution system.

---
## Virtual Environment Set Up
You will need to have python3 and anaconda installed. If these are not installed, please install them.
- https://www.python.org/downloads/
- https://www.anaconda.com/download/

### Open Terminal and the commands below to install the correct packages to run our code
`conda create -n hackathon2024env`

`conda activate hackathon2024env`

`pip install -r requirements.txt`

### To obtain the data, you will need
- All necessary data files can be downloaded from this link: https://drive.google.com/file/d/11uGTK2-DmquGXKvqoUxr0KA8ZDj8-wpV/view?usp=sharing
- Note: Please ensure that the folder structure is not changed or the code will not run

### File Structure (please run in this order)
- extract_data_from_pdfs.ipynb   # Jupyter Notebook for extracting data from PDFs
- extract_player_data.ipynb      # Jupyter Notebook for analyzing player data
- Player_Substitution.ipynb      # Jypyter Notebook for analysing substition player 
- initial_lineup_generator.py    # Python script for generating initial lineups
- requirements.txt               # List of required packages for the project
- README.md                      # Project description and guide
