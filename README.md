## Steps to Run the Code

1. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   
2. **Activate the virtual environment**:

   - On Windows:
      ```bash
      .\venv\Scripts\activate
      ```
   - On macOS and Linux:
      ```bash
      source venv/bin/activate
      ```
3. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Download and extract data**:

   - Get the data from the benign folder at [GitHub link](https://github.com/jzadeh/aktaion/tree/master/data/proxyData) and extract it to:
     ```bash
        ../data/raw_data_benign
     ```
   - Obtain data from the exploit folder and save it to:
   ```bash
        ../data/raw_data_exploit
     ```
   
5. **Run data extraction scripts**:
    ```bash
   cd ../data_extraction
   python data_extraction_begnin.py 
   
   python exploit_data_extractor.py
    ```
   
6. **Execute the main script**:
    ```bash 
       python main.py
    ```

7. **After the above step is complete Start the API**:
    ```bash 
       python api.py
    ```