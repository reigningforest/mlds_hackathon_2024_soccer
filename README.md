# Messi Data’s Playbook: Optimizing Lineups and Substitutions with AI

Team Messi Data: John He, Daeun Ji, Teddy Debreu, and Kavya Bhat

### Purpose
This project focuses on analyzing soccer match data and developing an AI-powered lineup recommendation and substitution system.

---

### File Structure
- `README.md`                      # Project description and guide
- `requirements.txt`               # List of required packages for the project
- `extract_data_from_pdfs.ipynb`   # Jupyter Notebook for extracting data from PDFs
- `extract_player_data.ipynb`      # Jupyter Notebook for analyzing player data
- `initial_lineup_generator.py`    # Python script for generating initial lineups
- `Player_Substitution.ipynb`      # Jypyter Notebook for analysing substition player 

---

# How to use our code
## Clone Repository
- Clone our repository into a folder. You will be working within this folder.

## Data Location
- All necessary data files can be downloaded from this link: https://drive.google.com/file/d/11uGTK2-DmquGXKvqoUxr0KA8ZDj8-wpV/view?usp=sharing
- Download and unzip the data
- Please drag folders: `match-reports`, `player-data`, and `team-data` into the cloned repository folder
- Note: Please ensure that you do not drag the parent folder `Hackathon Data` into the cloned repository folder

## Virtual Environment Set Up
You will need to have python3 and anaconda installed. If these are not installed, please install them.
- https://www.python.org/downloads/
- https://www.anaconda.com/download/

### Open Terminal and run the commands below to install the correct packages to run our code
`conda create -n hackathon2024env`

`conda activate hackathon2024env`

`conda install pip`

`pip install -r requirements.txt`

## Generate cleaned data
- Open `extract_data_from_pdfs.ipynb` and run all code chunks
- Open `extract_player_data.ipynb` and run all code chunks

## Run ML models
- Open and run (or run in terminal) `initial_lineup_generator.py`
- Open `Player_Substitution.ipynb` and run all code chunks

## Run AI Chat Bot
- Make sure you are in our directory.
- Get OpenAI API Key for GPT 4o-mini.
- ON MAC-Open terminal and write this command: `export OPENAI_API_KEY="your-key-here"`
- ON PC-Open Powershell and write this command: `$ENV:OPENAI_API_KEY="your-key-here"`
- Run Command: `python tedlasso.py` in terminal


