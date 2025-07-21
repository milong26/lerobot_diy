1. lerobot_diy目录下: 下载模型：wget -P models/objdection/yoloe https://hf-mirror.com/jameslahm/yoloe/resolve/main/yoloe-v8l-seg.pt 
2. git clone https://github.com/THU-MIG/yoloe.git
   
感覺yoloe不是很好用

还是用dino吧。

git clone https://github.com/IDEA-Research/GroundingDINO.git 代码已经上传到lerobot_diy里面了,可以不用关

然后pip install -e .

<!-- 把dino的权重放在lerobot_diy/models/objdection/dinoground/groundingdino_swint_ogc.pth这个目录 -->

在 models/objdection/dinoground文件夹里面下载
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

