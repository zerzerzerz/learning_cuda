# README
```bash
du --max-depth 1 -h
```
- 删除所有的exe文件，将stderr重定向，这样就算没有exe文件也不会报错
```bash
if ls *.exe > /dev/null 2>&1; then rm *.exe; fi
```