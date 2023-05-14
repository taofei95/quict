# vp-qcda

## Project setup
```
npm install
```

### Compiles and hot-reloads for development
```
npm run serve
```

### Compiles and minifies for production
```
npm run build
```

### Lints and fixes files
```
npm run lint
```

### Customize configuration
See [Configuration Reference](https://cli.vuejs.org/config/).


```
cd /home/wangzhanyu/quict/QuICT/UI/communication
conda run -n zhanyu2 --no-capture-output gunicorn -w 1 --threads 100 -b '0.0.0.0:8080' server:app
```