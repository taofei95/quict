<template>
  <el-container style="background-color: transparent; padding: 0px">
    <el-header>
      <el-row>
        <el-col :span="24" class="pro_title">Programing</el-col>
      </el-row>
    </el-header>
    <el-main style="padding: 0 15px; height: calc(100vh - 60px - 60px - 60px)">
      <el-input
        v-model="ProgramText"
        v-on:change="text_change"
        class="pro_zone"
        type="textarea"
      >
      </el-input>
    </el-main>
  </el-container>
</template>
<style>
.el-textarea__inner {
  height: calc(100vh - 60px - 60px - 60px - 20px) !important;
  width: 98% !important;
  background-color: transparent !important;
  color: azure !important;
  border-width: 2px 0 2px 0 !important;
  padding: 5px 5px !important;
}

.pro_title {
  font-family: MicrosoftYaHeiUI;
  color: #ffffff;
  text-align: left;
}
</style>
<script>
export default {
  props: {
    ProgramTextIn: String,
  },
  data: function () {
    return {
      tick: 0,
      updated: false,
      ProgramText: "",
      InCache: "",
    };
  },
  methods: {
    text_change(event) {
      this.tick = 0;
      this.updated = true;
      console.log(event);
    },
  },
  mounted: function () {
    this.ProgramText = this.ProgramTextIn;
    this.InCache = this.ProgramTextIn;
    setInterval(() => {
      this.tick += 1;
      // console.log(this.tick);
      // 当program text 变化时， 更新text area绑定的内容。
      if (this.ProgramTextIn != this.InCache) {
        this.InCache = this.ProgramTextIn;
        this.ProgramText = this.ProgramTextIn;
        this.updated = false;
        console.log("in", this.ProgramTextIn);
      }
      // 当text area绑定的内容变化后1秒，通知外层
      if (this.updated && this.tick >= 5) {
        this.$emit("ProgramUpdate", this.ProgramText);
        this.updated = false;
        console.log("out", this.ProgramText);
      }
    }, 200);
  },
  watch: {},
  emits: {
    ProgramUpdate: null,
  },
};
</script>