### Colon Cancer Classification using Convolutional Neural Networks
This project aims to classify images in the given dataset as colon adenocarcinoma or benign colon tissue, making use of convolutional neural networks (CNN) and the PyTorch library.

#### Overview:

Currently, diagnostic measures diagnosing colon cancer involves a relatively simple procedure called a colonoscopy. While physicians are generally able to extract potentially cancerous polyps, they cannot tell simply by looking at them whether or not the polyps are cancerous without sending them to the lab for further analysis. It is in the lab where errors can be made when determining the cancerous nature of polyps, since certain polyps may be mistakenly overlooked. In this regard, machine learning holds immense promise for improving the screening efforts of physicians to prevent colorectal cancer cases and improve early-detection efforts. By training a convolutional neural network on existing lab samples of colon tissue, we can hope to accurately identify polyp samples (which are simply extensions of tissue) when they reach the lab. The below graphic contains 16 labelled image samples that were used in the training of the CNN model. "colon-aca" and "colon-n" represent colon adenocarcinoma and benign colon tissue, respectively.

<img src="assets/imggal.png" alt="Figure1" width="400"/>

#### Results:
- Various model architectures provided different accuracy on test data
    - Random Forest: 53.3%
    - CNN: 66.7%
    - VGG-16 + Random Forest: 95.6%
    - YOLOv8 + Random Forest: 96.2%

#### Data & Data Processing:

- 10000 images of colon adenocarcinoma and benign colon tissue
    - training data / validation data split = 80/20
- Various transforms were applied to the image data for better generalization
    - resize to 180 x 180 pixels
    - random horizontal flip
    - random rotation by 10 degrees
    - transform to tensor
    - normalize tensor data

#### Try it:

Refer to the [report](report.pdf) and [code](./Code) for further details and analysis or try uploading your own samples [here](https://www.gradio.app/guides/quickstart)

#### Acknowledgments:

The dataset used in this project comes from:

Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). arXiv:1912.12142v1 [eess.IV], 2019
