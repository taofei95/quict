import { createApp } from 'vue'
import App from './App.vue'
import ElementPlus from 'element-plus';
import 'element-plus/lib/theme-chalk/index.css';
import 'xterm/css/xterm.css'
import { io } from "socket.io-client";


function uuid() {
  var s = [];
  var hexDigits = "0123456789abcdef";
  for (var i = 0; i < 36; i++) {
    s[i] = hexDigits.substr(Math.floor(Math.random() * 0x10), 1);
  }
  s[14] = "4"; // bits 12-15 of the time_hi_and_version field to 0010
  s[19] = hexDigits.substr((s[19] & 0x3) | 0x8, 1); // bits 6-7 of the clock_seq_hi_and_reserved to 01
  s[8] = s[13] = s[18] = s[23] = "-";

  var uuid = s.join("");
  return uuid;
}

const app = createApp(App);
app.config.globalProperties.background = 'http://localhost:5000';
app.config.globalProperties.uuid = uuid();
app.use(ElementPlus);

app.use(io);
app.config.globalProperties.socket = io(`${app.config.globalProperties.background}/api/pty`);
app.config.globalProperties.socket.on('connect', () => {
  console.log('socket.io connected.')
  app.config.globalProperties.connected = true
});

app.config.globalProperties.socket.on('disconnect', () => {
  console.log('socket.io disconnected.')
  app.config.globalProperties.connected = false
});
app.mount('#app')
