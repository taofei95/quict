<template>
  <el-container style="background-color: #13141c; padding: 0px">
    <el-header style="height: 50px">
      <!-- <el-row>
      <el-col :span="12"> -->
      <el-space style="height: 50px; font-size: var(--el-font-size-large); width: 80%" size="large"
        direction="horizontal">
        <div
          style="background-image: url('/assets/logo.png'); background-repeat: no-repeat; background-position: left; width: 160px;  height: 45px;">
        </div>

        <span class="span_selected" id="span_QuCompuser" @click="SelectPageQuCompuser">
          QuCompuser
        </span>

        <span class="span_not_selected" id="span_QCDA" @click="SelectPageQCDA">
          QCDA
        </span>
      </el-space>
      <el-space style="height: 50px; font-size: var(--el-font-size-large); width: 20%" size="large"
        direction="horizontal">
        <span class="span_not_selected" id="span_QCDA" @click="dialogUser = true">
          User
        </span>
        <span class="span_not_selected" id="span_QCDA" @click="Logout">
          Logout
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

      <el-dialog title="User" v-model="dialogUser" width="30%" :close-on-click-modal="false"
        :close-on-press-escape="false" :show-close="false">
        <el-space direction>
          <div>USER: {{ user }}</div>
          <div>E-Mail: {{ email }}</div>
        </el-space>
        <template #footer>
          <span class="dialog-footer">
            <el-button type="primary" @click="dialogUser = false">OK</el-button>
            <el-button type="primary" @click="dialogUser = false; dialogPsw = true">Change Password</el-button>
            <el-button type="primary" @click="Unsubscribe()">Unsubscribe</el-button>
          </span>
        </template>
      </el-dialog>

      <el-dialog title="Change Password" v-model="dialogPsw" width="30%" :close-on-click-modal="false"
        :close-on-press-escape="false" :show-close="false">
        <label>Old<el-input v-model="reg_psw" type="password" show-password></el-input></label>
        <label>New<el-input v-model="reg_psw" type="password" show-password></el-input></label>
        <label>Confirm<el-input v-model="reg_psw" type="password" show-password></el-input></label>

        <template #footer>
          <span class="dialog-footer">
            <el-button type="primary" @click="ChangePsw()">OK</el-button>
            <el-button type="primary" @click="dialogPsw = false">Cancel</el-button>
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
      dialogUser: false,
      dialogPsw: false,
      user: "",
      psw: "",
      email: "",
      reg_user: "",
      reg_psw: "",
      reg_email: "",
      for_user: "",
      for_email: "",
      ch_psw: "",
      ch_psw_2: "",
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
    Logout() {
      this.socket.emit("logout", {
        uuid: this.uuid,
        content: {
          user: this.user,
        },
      });
      this.dialogLogin = true;
    },
    User() {
      this.dialogUser = true;
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
    ChangePsw() {
      if (this.ch_psw != this.ch_psw_2) {
        this.ShowError('Password not the same.')
      }
      else {
        this.socket.emit("changepsw", {
          uuid: this.uuid,
          content: {
            user: this.user,
            new_password: this.ch_psw
          },
        });
      }
    },
    Unsubscribe() {
      this.$confirm('Do you want to Unsubscribe?', 'Confirm', {
        confirmButtonText: 'OK',
        cancelButtonText: 'Cancel',
        type: 'warning'
      }).then(() => {
        this.socket.emit("unsubscribe", {
          uuid: this.uuid,
          content: {
            user: this.user,
          },
        });
      }).catch(() => {

      });
    },
    ShowError(msg) {
      this.$message({
        showClose: true,
        message: msg,
        type: 'error'
      });
    },
    ShowOK(msg) {
      this.$message({
        showClose: true,
        message: msg,
        type: 'success'
      });
    }
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
      this.email = content['info'][1]
      this.socket.emit("get_gate_set", { uuid: this.uuid, source: "Home" });
    });

    this.socket.on("need_login", (content) => {
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.dialogLogin = true;
    });

    this.socket.on("register_ok", (content) => {
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.dialogRegister = false;
      this.ShowOK("Register success. Please login.")
      this.socket.emit("testLogin", { uuid: this.uuid });
    });

    this.socket.on("forget_ok", (content) => {
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.dialogForget = false;
      this.ShowOK("Password reseted, Please check your e-mail box.")
      this.socket.emit("testLogin", { uuid: this.uuid });
    });

    this.socket.on("update_psw_ok", (content) => {
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.dialogPsw = false;
      this.ShowOK("Password changed.")
      this.socket.emit("testLogin", { uuid: this.uuid });
    });

    this.socket.on("unsubscribe_ok", (content) => {
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.dialogUser = false;
      this.ShowOK("Unsubscribed success.")
      this.socket.emit("testLogin", { uuid: this.uuid });
    });

    this.socket.on("login_error", (content) => {
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.ShowError("Login failed.")
      this.socket.emit("testLogin", { uuid: this.uuid });
    });
  },
  watch: {},
};
</script>