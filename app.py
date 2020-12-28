from flask import Flask, request, send_file
from datetime import datetime
from flask_cors import CORS
from flask_ngrok import run_with_ngrok
import numpy as np
import base64
import io
from PIL import Image
from argparse import ArgumentParser

import os

def generate_links(images):
    data = np.load(app.config['lp'], allow_pickle=True).tolist()
    links = []
    for img in images:
        # wrong file name so we need prefix /content/content
        try:
          links.append(
            {
              'img_name':img,
              'film_link':data['/content/content/storage/{}'.format(img.split('/')[-1])]
            }
            )
        except KeyError:
            links.append(
            {
              'img_name':img,
              'film_link':''
            }
            )
    return links


def get_encoded_img(image_path):
    img = Image.open(image_path, mode='r')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return my_encoded_img


def save_image(file):
    q_id = datetime.timestamp(datetime.now())
    q_id = str(int(q_id))
    print(os.path.join(os.getcwd(), app.config['qp'], q_id, file.filename))
    cmd = "mkdir -p {}".format(os.path.join(app.config['qp'], q_id))
    os.system(cmd)
    file.save(os.path.join(os.getcwd(), app.config['qp'], q_id, file.filename))
    return q_id


def query(query_id, method="COLOR", similarity="cosine", lsh=0):
    query_id = str(query_id)
    base = '/content'
    input_p = os.path.join(base, app.config['qp'], query_id)
    try:
        os.mkdir("{}_out".format(input_p))
    except FileExistsError:
        print("{}_out is exists".format(input_p))

    feature_path = os.path.join(app.config['fp'], method)

    cmd = """python main.py \
    --option=query \
    --input_path={ip} \
    --output_path={ip}_out \
    --feature_path={fp} \
    --feature_method={method} \
    --similarity_measure={similarity} \
    --LSH={lsh}""".format(ip=input_p, fp=feature_path, method=method, lsh=lsh, similarity=similarity)
    print(method, similarity, lsh)
    print(feature_path)
    print(cmd)
    os.system(cmd)

    return "{ip}_out".format(ip=input_p)


def tojson(output_path):
    out = os.listdir(output_path)
    out = list(filter(lambda x: 'npz' in x, out))
    print(out)
    data = np.load(os.path.join(output_path, out[0]))
    ranks = data['ranks']
    paths = data['paths']
    print(ranks)
    ps = []
    for rank in ranks[0]:
        ps.append(paths[rank])
    return ps


#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————#

app = Flask(__name__)
def allowed_file(filename):
    return filename.split('.')[-1] in ALLOWED_EXTENSIONS and len(filename.split('.')) >= 2


@app.route('/', methods=['GET'])
def html():
    f = open('web/index.html')
    data = f.read()
    return data


@app.route('/api', methods=['POST', 'GET'])
def index():
    t = {"message": "Welcom to Anime Finder !"}
    print(app.config['fp'])
    print(app.config['lp'])

    # print(t)
    return t, 200


@app.route('/electronics.png')
def logo():
    return send_file(os.path.join(os.getcwd(), 'electronics.png'), as_attachment=True), 200


@app.route('/api/query', methods=['post'])
def api_query():
    if request.files:
        similarity = request.form.get('similarity', 'cosine')
        method = request.form.get('method', 'COLOR')
        lsh = request.form.get('lsh', '0')
        # print(similarity, method, lsh)
        file = request.files['query_image']
        if allowed_file(file.filename) and file:
            q_id = save_image(file)
            print(q_id)
            out = query(q_id, method=method, similarity=similarity, lsh=lsh)
            out = tojson(out)
            out = list(map(lambda x: x.split('/')[-1], out))
            data = generate_links(out)
            print(data)
        # return {"img": [], "links": [], "q_id": q_id}, 200

        return {"data":data, "q_id": q_id}, 200
    else:
        return {"message": "No file"}, 200


@app.route('/api/image/q/<string:q_id>', methods=['GET'])
def get_q_image(q_id):
    fp = os.path.join( app.config['qp'], q_id)
    if os.path.exists(fp):
        img = os.listdir(fp)[0]
        fp = os.path.join(fp, img)
        encoded = get_encoded_img(fp)
        return {"img": encoded}, 200
    else:
        return {'error': 'Query image not found'}, 404


@app.route('/api/image/<string:img_name>', methods=['GET'])
def get_image(img_name):
    base = '/content/database'
    fp = os.path.join(base, img_name)
    if os.path.exists(fp):
        encoded = get_encoded_img(fp)
        return {"img": encoded}, 200
    else:
        return {'error': 'Image not found'}, 404


if __name__ == '__main__':
    os.system('mkdir -p /content/queries')
    parser = ArgumentParser()
    parser.add_argument('-ng','--ngrok',default=0)
    parser.add_argument('-fp','--feature_path',default='/content/drive/MyDrive/Information_Retrieval/perfect_feature_image/',help='Path to your feature folder, must be in our format')
    parser.add_argument('-lp','--link_path',default='/content/drive/MyDrive/Official_OnePiece_images/Link_OnePiece.npy',help='Path to your image_2_film_link file (.npz), must be in our format')
    parser.add_argument('-db','--database',default='/content/database',help='Path to your images database, use for return image, must be in our format')

    args = parser.parse_args()
    args = vars(parser.parse_args())
    app.config['fp'] = args['feature_path']
    app.config['lp'] = args['link_path']
    ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
    app.config['qp'] = '/content/queries'
    CORS(app)
    if str(args['ngrok']) == '1':
        run_with_ngrok(app)
        print('ngrok')
    app.run()
