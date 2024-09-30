# GOOGLE PROJECT

## 일부 프로젝트만 사용하기 

```
git clone --no-checkout https://github.com/sepengsu/googleMLB5.git
```
```
cd googleMLB5 # Clone한 폴더에 이동 
```

```
git sparse-checkout init --cone
```
```
git sparse-checkout set set project1 # 프로젝트 폴더 지정 
```
```
git checkout main
```

이후 git 폴더를 삭제하세요 

