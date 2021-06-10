from flask import Flask, request,jsonify
from flask_restful import Resource
from flask_jsonpify import jsonpify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from datetime import datetime
import pandas as pd
import numpy as np
import requests
import json
import datetime


def convertTimeStampToUTC(data):
  from datetime import datetime
  return datetime.fromtimestamp(data)
  
def getURL(apiKey,cryptoSymbol,frequency,timestamp,MaxRecordsToGetOnEachCall):
  API_KEY='&api_key='+str(apiKey)
  limit='&limit='+str(MaxRecordsToGetOnEachCall)
  cryptoHeader='fsym='+str(cryptoSymbol)
  ouputCurrency='&tsym=USD'

  if(timestamp==""):
    url='https://min-api.cryptocompare.com/data/v2/histo'+frequency+'?'+cryptoHeader
  else:  
    url='https://min-api.cryptocompare.com/data/v2/histo'+frequency+'?'+cryptoHeader+'&toTs='+str(timestamp)

  return url+ouputCurrency+API_KEY+limit  

def getCryptoData(apikey,crypto,frequency,numOfTimesToHitApi,MaxRecordsToGetOnEachCall):
  numOfTimesToHitApi=5
  data=pd.DataFrame()
  nextTimeStamp=0

  for i in range(0,numOfTimesToHitApi):
    if(nextTimeStamp==0):
      url=getURL(apikey,crypto,frequency,"",MaxRecordsToGetOnEachCall)
    else:
      url=getURL(apikey,crypto,frequency,nextTimeStamp,MaxRecordsToGetOnEachCall)

    response = requests.get(url).json()
    partialData=pd.DataFrame(response.get('Data').get('Data'))
    nextTimeStamp=partialData['time'].min()-1
    data=pd.concat([data,partialData])


  data=data.sort_values(by=['time'])
  data['timeUTC']=data['time'].apply(convertTimeStampToUTC)
  data=data.reset_index(drop=True)
  return data

def GRU(cryptoData):
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM,GRU
    from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
    from sklearn.metrics import mean_squared_error
    from pandas import Series
    
    data=cryptoData
    data=data.set_index(pd.DatetimeIndex(data['timeUTC']))['close']
    
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)
    look_back=3
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i+look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)
    
    numpy.random.seed(0)
    dataframe = data
    dataset = dataframe.values
    dataset = dataset.astype('float64').reshape(-1, 1)
    
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    from keras.layers import Activation, Dense,Dropout
    model = Sequential()
    model.add(LSTM(256, return_sequences=True,input_shape=(1, look_back)))
    model.add(LSTM(256))
    model.add(Dense(1))
    import keras
    from keras import optimizers
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=50, verbose=1,shuffle=False,batch_size=50)
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    predictions = numpy.empty_like(dataset)
    predictions[:, :] = numpy.nan
    predictions[look_back:len(trainPredict)+look_back, :] = trainPredict
    predictions[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    predictionsDF=pd.DataFrame(predictions,columns=["predicted"],index=dataframe.index)
    ans=pd.concat([dataframe,predictionsDF],axis=1)
    return ans

def customcurrency(ans):
    hit_ans=ans.dropna()
    ref_sample=hit_ans.tail(1)
    ref_close=ref_sample['close']
    close_value=pd.Series(ref_close).values[0]
    ref_margin=close_value/10
    close_list=hit_ans['close'].values.tolist()
    predicted_list=hit_ans['predicted'].values.tolist()
    hit=0
    for i in range(len(close_list)):
        if predicted_list[i]>=(close_list[i]-ref_margin) and predicted_list[i]<=(close_list[i]+ref_margin):
            hit+=1
    custom_accuracy=hit/len(close_list)*100
    ref_margin=close_value/100
    hit=0
    for i in range(len(close_list)):
        if predicted_list[i]>=(close_list[i]-ref_margin) and predicted_list[i]<=(close_list[i]+ref_margin):
            hit+=1
    custom_accuracy=hit/len(close_list)*100
    return custom_accuracy

@app.route('/getdata',methods=['POST'])
def get_data():
    currencies={1:"ETH",2:"BTC",3:"DOGE",4:"BNB",5:"XRP",6:"DOT",7:"ADA",8:"BUSD",9:"MATIC",10:"LTC"}
    frequencies={1:"day",2:"hour",3:"minute"}
    request_data = request.get_json()
    n = request_data["cryptocurrency"]
    freq = request_data["frequency"]
    API_KEY=""
    crypto=currencies[n]
    numOfTimesToHitApi=5
    MaxRecordsToGetOnEachCall=200
    frequency=frequencies[freq]
    cryptoData=getCryptoData(API_KEY,crypto,frequency,numOfTimesToHitApi,MaxRecordsToGetOnEachCall)
    df_list = cryptoData.values.tolist()
    JSONP_data = jsonpify(df_list)
    return JSONP_data

@app.route('/getprediction',methods=['POST'])
def get_prediction():
    currencies={1:"ETH",2:"BTC",3:"DOGE",4:"BNB",5:"XRP",6:"DOT",7:"ADA",8:"BUSD",9:"MATIC",10:"LTC"}
    frequencies={1:"day",2:"hour",3:"minute"}
    request_data = request.get_json()
    n = request_data["cryptocurrency"]
    freq = request_data["frequency"]
    API_KEY=""
    crypto=currencies[n]
    numOfTimesToHitApi=5
    MaxRecordsToGetOnEachCall=200
    frequency=frequencies[freq]
    cryptoData=getCryptoData(API_KEY,crypto,frequency,numOfTimesToHitApi,MaxRecordsToGetOnEachCall)
    cryptoData['PriceClose2D']=cryptoData['close']
    shift=2
    cryptoData['PriceClose2D']=cryptoData['PriceClose2D'].shift(-shift)
    cryptoData=cryptoData.fillna(method='ffill')
    ans = GRU(cryptoData)
    ans=ans.dropna()
    # ans = ans.replace({np.nan: None})
    timeUTC = ans.index.values.tolist()
    close = ans["close"].values.tolist()
    predicted = ans["predicted"].values.tolist()
    # JSONP_data = jsonpify(df_list)
    # d = ans.to_dict()
    # j = json.dumps(d)
    # def myconverter(o):
    #  if isinstance(o, datetime.datetime):
    #     return o.__str__()
    # my_json_data = json.dumps(my_dictionary, default=convert_timestamp)
    # ans.reset_index().values.tolist()
    # custom_currency = {"custom_currency":customcurrency(ans),"ans":{"time": timeUTC,"close":close,"predicted":predicted}}
    c=float("{:.2f}".format(customcurrency(ans)))
    result = {"ratio":c,"ans":ans.reset_index().values.tolist()}
    return result

app.run(port=5000,debug=True)