from install_dependencies import install_dependencies
from preprocessing import preprocess_data
from models.feature_selection import feature_selection as feature_selection_main

install_dependencies()

preprocess_data()

feature_selection_main()