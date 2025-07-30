# 使用官方的 Python 3.9 slim 版本作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 将依赖文件复制到工作目录
COPY requirements.txt requirements.txt

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 将项目代码复制到工作目录
COPY . .

# 设置环境变量，让 Flask/Gunicorn 知道如何运行你的应用
# 格式是 文件名:Flask应用实例名
ENV APP_MODULE="main:app"

# 容器对外暴露的端口，需要和 Cloud Run 的设置一致
# $PORT 这个变量会由 Cloud Run 自动提供
ENV PORT 8080

# 容器启动时运行的命令
# 使用 gunicorn 作为生产环境的 WSGI 服务器
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--threads", "8", "--timeout", "0", "main:app"]