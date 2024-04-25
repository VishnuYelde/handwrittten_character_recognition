# Handwritten Character Recognition

This project is a handwritten character recognition system implemented using Python, Flask, OpenCV, NumPy, and Keras. It allows users to upload images containing handwritten characters, and the system predicts the characters present in the images using a trained deep learning model.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- Upload images containing handwritten characters.
- Predict the characters present in the images.
- Supports grayscale images.
- Easy-to-use web interface.


## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/handwritten-character-recognition.git
    ```

2. Navigate to the project directory:

    ```bash
    cd handwritten-character-recognition
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Download the pre-trained model file and place it in the `model` directory.

5. Run the Flask application:

    ```bash
    python app.py
    ```

6. Open a web browser and go to `http://localhost:5000` to access the application.

## Usage

1. Open the web application in a browser.
2. Upload an image containing a handwritten character.
3. Click the "Upload" button to process the image.
4. The predicted character will be displayed on the screen.

## Contributing

Contributions are welcome! Here's how you can contribute to this project:

- Fork the repository.
- Create a new branch (`git checkout -b feature-branch`).
- Make your changes.
- Commit your changes (`git commit -am 'Add new feature'`).
- Push to the branch (`git push origin feature-branch`).
- Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
