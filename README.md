# Kargin Summarization: Revolutionizing Effortless Article Summarization

**Authors:** Areg Petrosyan, Artyom Khachatryan, Gagik Mkrtchyan, Samvel Karapetyan  
**Contact:** acafinalproject@gmail.com

Welcome to the unveiling of our team's final project, "Kargin Summarization," a cutting-edge solution for text summarization. For seamless execution of our codebase, kindly adhere to the following instructions:

You can download our checkpoints from [drive](https://drive.google.com/drive/folders/1RH_GFBDfiRDq3K_4VQ4YIyr6o_zhv03f?usp=drive_link)

1. **Setting Up the Environment:**
   
   Run the following command to create a new environment:

   ```
   conda env create
   ```

2. **Configuration and Execution:**

   Configure the `dotenv` files and execute the provided Python script:

   ```
   python train.py
   ```

3. **Harnessing the Potential of Streamlit:**

   Once the training is complete (or even without training), leverage the capabilities of the Streamlit package for result inference:

   ```
   streamlit run app.py
   ```

### Unleashing the Power of Kargin Summarization

In the context of this project, we have devised two novel networks. These networks have been trained from scratch, utilizing distinct datasets. Furthermore, we have integrated a complementary approach by fine-tuning on the BART network architecture. For a deeper dive into the details, we encourage you to explore our presentation slides.html. 