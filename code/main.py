from install_dependencies import install_dependencies
from preprocessing import preprocess_data
from models.feature_selection import feature_selection as feature_selection_main
from models.run_fuzzy_topsis import run_fuzzy_topsis_feature_selection
from models.run_dgwo import run_dgwa_feature_selection
# install_dependencies()
#
# preprocess_data()
#
# feature_selection_main()

# run_fuzzy_topsis_feature_selection()

run_dgwa_feature_selection()