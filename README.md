# mlai-fewshot

```
cd code/ours
pip install -r requirements.txt
```

### Train mini-imagenet
```
cd code/ours
./run-mini.sh
```

### Train tiered-imagenet
```
cd code/ours
./run-tiered.sh
```

### Train omniglot
```
cd code/ours
./run-omni-rot.sh
```

### Use Tensorboard
```
tensorboard --dir runs
```

Data는 materials폴더에 넣고 사용하시면 됩니다.
서버에서 데이터 경로: /st1/hayeon/materials

save 폴더: 스냅샷 저장
log 폴더: txt파일로 로그 저장

