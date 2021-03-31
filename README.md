# ADNI_DataAnalysis
Data analysis for ADNI CN/AD prediction and MCI longitudinal follow up.

Part 1 contains the modeling for the probability of AD (Alzheimer's Disease) considering only the 229 CN (cognitively normal) and 188 AD patients. The dataset has 579 lipidomic and 116 MRI brain volume features. The best model is selected via cross validation and used to predict the probability of AD at baseline for the 402 MCI (mild cognitive impairment) patients. The probability of AD for the MCI patients is also binned into 5 bins spaced by 0.2. 

Part 2 includes the longitudinal follow up analysis for the MCI patients. It examines whether MCI patients in higher bins have a worse cognitive outcome in terms of executive functioning (ADNI-EF), memory (ADNI-MEM), language (ADNI-LAN),visuospatial functioning (ADNI-VS), and the number of errors on the Mini Mental State Examination (MMSE). 

View Part 1: https://htmlpreview.github.io/?https://github.com/rokapre/ADNI_DataAnalysis/blob/main/ADNI_ML_nb.jl.html

View part 2: https://htmlpreview.github.io/?https://github.com/rokapre/ADNI_DataAnalysis/blob/main/ADNI_MCI_FollowUp.html