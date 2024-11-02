# gestures recognition

## 说明
为了保证服务可以顺利调用摄像头，请使用https协议，在 Flask 环境下使用 HTTPS，可以通过以下方法实现：
1.生成私钥和证书：
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365
2. 服务使用生成私钥和证书启动服务
app.run(host='0.0.0.0', port=443, ssl_context=('cert.pem', 'key.pem'))
