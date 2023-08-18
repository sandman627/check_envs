


# Installation 

## python
python3.8-dev 혹은 python3.9-dev 가 필요하다.
유념할 것은 dev 버전이 필요하다는 것.



## FastDownWard


```
sudo apt install cmake g++ git make python3
```


## ai2thor
Unity3D를 기반으로 만든 환경이다.
버전은 2.1.0 을 사용해야한다.
```
pip install ai2thor==2.1.0
```





## Alfred
https://github.com/askforalfred/alfred
Ai2thor 환경을 바탕으로 만든 benchmark다. 




## ALFWorld

https://github.com/alfworld/alfworld
위의 Alfred의 한계점을 Textworld를 덧붙여서 극복한 환경이다.
ALFWorld를 실행하기 위해서는 추가 data가 필요하다. 더불어 해당 data는 config.yaml로 사용될 것이기에 폴더의 경로를 환경변수로 추가해준다.



```
git clone https://github.com/alfworld/alfworld.git alfworld

cd alfworld

pip install -r requirements.txt

pip install .
```

```
export ALFWORLD_DATA=<storage_path>
alfworld-download
```


### pytorch, torchvision 버전 호환 문제
_lzma 가 없다는 error가 나타나기도 한다.








추가로 ALFWorld의 Environment를 실행하려면, 아래에 보이듯이 alfworld/configs/base_config.yaml 파일을 사용해야 한다.
현재 기본으로 내장되어 있지만 혹시 문제가 생기면 아래 위치를 찾아가자

```
python train.py alfworld/configs/base_config.yaml 
```




### Textworld 

```    
    gamefiles
    batch_size
    request_infos
    _gamefiles_iterator
    batch_env
    action_space
    observation_space
    spec
    last_commands
    ob
```