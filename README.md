# AI-Detector

Codes of our AI-Detector method

### ExampleData
- Data used in experiments
- **raw.txt:** Raw data of human-authored titles and abstracts
- **exp-data.txt:** The constructed AI-generated text detection dataset
- **exp-data-AID.txt:** The constructed detection features used in AI-Detector
- **word_entropy.json:** The domain-aware word information entropy look-up table

### Crawerl
- **Crawerl-Data.ipynb:** Crawerl codes for constructing GPT-generated detection datasets
- **Crawerl-AIDetector.ipynb:** Crawerl codes for generating AI-summarized versions of the target texts


### Detection
- **ParseFeature.ipynb:** Codes for generating detection features used in our AI-Detector method
- **Detection.ipynb:** Codes for training the detection model of our AI-Detector method