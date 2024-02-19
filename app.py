from flask import Flask, request, jsonify
from imageSimilarity import compare_images
import requests

app = Flask(__name__)

def download_image(image_url, filename):
    response = requests.get(image_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open a file in binary write mode
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Image successfully downloaded and saved as {filename}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")


@app.route('/imageSimilarity', methods=['POST'])
def handle_post():
    data = request.json 
    download_image(data['url1'], 'image1.jpg')
    download_image(data['url2'], 'image2.jpg')

    return jsonify({"similarity": compare_images('image1.jpg', 'image2.jpg')}), 200

if __name__ == '__main__':
    app.run(debug=True)