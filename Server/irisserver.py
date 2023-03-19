from flask import Flask,request,jsonify
import irisutil



iris = Flask(__name__)

@iris.route('/',methods=['GET'])
def dummy():
    response=jsonify({})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@iris.route('/get_result', methods=['GET','POST'])
def get_result():
    sl = float(request.form["sepal_length"])
    sw = float(request.form["sepal_width"])
    pl = float(request.form["petal_length"])
    pw = float(request.form["petal_width"])

    response = jsonify({
        'prediction': irisutil.result(sl,sw,pl,pw)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == '__main__':
    print("starting python flask server for prediction....")
    irisutil.load_saved_artifacts()
    iris.debug=True
    iris.run()




