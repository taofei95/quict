<template>
  <el-container style="background-color: #13141c; padding: 0px">
    <el-header style="height: 50px">
      <!-- <el-row>
      <el-col :span="12"> -->
      <el-space style="height: 50px; font-size: var(--el-font-size-large); width: 100%" size="large"
        direction="horizontal">
        <div style="
            background-image: url('/assets/logo.png');
            background-repeat: no-repeat;
            background-position: left;
            width: 160px;
            height: 45px;
          "></div>

        <span class="span_selected" id="span_QuCompuser" @click="SelectPageQuCompuser">
          QuCompuser
        </span>

        <span class="span_not_selected" id="span_QCDA" @click="SelectPageQCDA">
          QCDA
        </span>
      </el-space>
      <!-- </el-col>
      </el-row> -->
    </el-header>
    <el-main class="page_zone">
      <QuCompuser class="page_selected" id="page_QuCompuser"></QuCompuser>
      <QCDA class="page_not_selected" id="page_QCDA"></QCDA>
      <el-dialog title="Login" v-model="dialogLogin" width="30%" :close-on-click-modal="false"
        :close-on-press-escape="false" :show-close="false">
        <label>USER<el-input v-model="user"></el-input></label>
        <label>PASSWORD<el-input v-model="psw" type="password" show-password></el-input></label>
        <template #footer>
          <span class="dialog-footer">
            <el-button type="primary" @click="login()">OK</el-button>
            <el-button type="primary" @click="Go2Register()">Register</el-button>
            <el-button type="primary" @click="Go2Forget()">Forget</el-button>
          </span>
        </template>
      </el-dialog>
      <el-dialog title="Register" v-model="dialogRegister" width="30%" :close-on-click-modal="false"
        :close-on-press-escape="false" :show-close="false">
        <label>USER<el-input v-model="reg_user"></el-input></label>
        <label>E-Mail<el-input v-model="reg_email"></el-input></label>
        <label>PASSWORD<el-input v-model="reg_psw" type="password" show-password></el-input></label>
        <template #footer>
          <span class="dialog-footer">
            <el-button type="primary" @click="Register()">OK</el-button>
            <el-button type="primary" @click="Back2Login()">Cancel</el-button>
          </span>
        </template>
      </el-dialog>
      <el-dialog title="Forget" v-model="dialogForget" width="30%" :close-on-click-modal="false"
        :close-on-press-escape="false" :show-close="false">
        <label>USER<el-input v-model="for_user"></el-input></label>
        <label>E-Mail<el-input v-model="for_email"></el-input></label>

        <template #footer>
          <span class="dialog-footer">
            <el-button type="primary" @click="Forget()">OK</el-button>
            <el-button type="primary" @click="Back2Login()">Cancel</el-button>
          </span>
        </template>
      </el-dialog>
    </el-main>
  </el-container>
</template>
<style>
.span_selected {
  color: blanchedalmond;
}

.span_not_selected {
  color: dimgray;
}

.page_zone {
  height: calc(100vh - 50px);
}

.page_selected {
  display: block;
}

.page_not_selected {
  display: none;
}

.page_zone {
  padding: 0px !important;
}
</style>

<script>
import QuCompuser from "./QuCompuser.vue";
import QCDA from "./QCDA.vue";
import * as d3 from "d3";

export default {
  props: {},
  data: function () {
    return {
      dialogLogin: false,
      dialogRegister: false,
      dialogForget: false,
      user: "",
      psw: "",
      reg_user: "",
      reg_psw: "",
      reg_email: "",
      for_user: "",
      for_email: "",
      CurrentPage: "QuCompuser",

      AllPages: ["QuCompuser", "QCDA"],
    };
  },
  components: {
    QuCompuser,
    QCDA,
  },
  methods: {
    SelectPageQuCompuser() {
      if (this.CurrentPage != "QuCompuser") {
        this.CurrentPage = "QuCompuser";
        d3.select("#span_QuCompuser").attr("class", "span_selected");
        d3.select("#span_QCDA").attr("class", "span_not_selected");
        d3.select("#page_QuCompuser").attr("class", "page_selected");
        d3.select("#page_QCDA").attr("class", "page_not_selected");
      }
    },
    SelectPageQCDA() {
      if (this.CurrentPage != "QCDA") {
        this.CurrentPage = "QCDA";
        d3.select("#span_QuCompuser").attr("class", "span_not_selected");
        d3.select("#span_QCDA").attr("class", "span_selected");
        d3.select("#page_QuCompuser").attr("class", "page_not_selected");
        d3.select("#page_QCDA").attr("class", "page_selected");
      }
    },

    login() {
      this.socket.emit("login", {
        uuid: this.uuid,
        content: {
          user: this.user,
          psw: this.psw,
        },
      });
    },
    testLogin() {
      this.socket.emit("testLogin", {
        uuid: this.uuid,
        content: {},
      });
    },
    Go2Register() {
      this.dialogLogin = false;
      this.dialogRegister = true;
    },
    Back2Login() {
      this.dialogRegister = false;
      this.dialogForget = false;
      this.dialogLogin = true;
    },
    Register() {
      this.socket.emit("register", {
        uuid: this.uuid,
        content: {
          user: this.reg_user,
          psw: this.reg_psw,
          email: this.reg_email,
        },
      });
    },
    Go2Forget() {
      this.dialogLogin = false;
      this.dialogForget = true;
    },
    Forget() {
      this.socket.emit("forget", {
        uuid: this.uuid,
        content: {
          user: this.for_user,
          email: this.for_email,
        },
      });
    },
  },
  mounted: function () {
    d3.select("#page_QCDA").attr("class", "page_not_selected");
    this.socket.emit("testLogin", { uuid: this.uuid });
    this.socket.on("login_success", (content) => {
      // 收到后端处理好的qasm，显示到前端qasm编辑区域
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.dialogLogin = false;
      this.socket.emit("get_gate_set", { uuid: this.uuid, source: "Home" });
    });

    this.socket.on("need_login", (content) => {
      // 收到后端处理好的qasm，显示到前端qasm编辑区域
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.dialogLogin = true;
    });

    this.socket.on("register_ok", (content) => {
      // 收到后端处理好的qasm，显示到前端qasm编辑区域
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.dialogRegister = false;
      this.socket.emit("testLogin", { uuid: this.uuid });
    });

    this.socket.on("forget_ok", (content) => {
      // 收到后端处理好的qasm，显示到前端qasm编辑区域
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.dialogForget = false;
      this.socket.emit("testLogin", { uuid: this.uuid });
    });
  },
  watch: {},
};
</script>