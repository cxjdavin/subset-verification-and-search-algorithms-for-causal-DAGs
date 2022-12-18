# Clone dct-policy repository
git clone https://github.com/csquires/dct-policy.git
cd dct-policy

python3 -m venv venv
source venv/bin/activate
pip install wheel
pip install networkx seaborn tqdm ipdb p_tqdm
pip install causaldag==0.1a135

# Grab PADS source files
wget -r --no-parent --no-host-directories --cut-dirs=1 http://www.ics.uci.edu/\~eppstein/PADS/

# Copy our modifications into experimental folder
cp ../our_code/*.py .
cp ../our_code_baseline/*.py baseline_policies
mv pdag.py venv/lib/python3.8/site-packages/causaldag/classes/

# Run experiments to obtain plots
python3 random_T_experiments.py
python3 exp_p0001_hop1.py
python3 exp_p0001_hop3.py

# Return to parent directory
cd ..

