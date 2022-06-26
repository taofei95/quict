<template>
  <el-container style="
      height: calc(100vh - 50px);
      font-size: var(--el-font-size-large);
      width: 100%;
    ">
    <el-header style="height: 50px">
      <el-steps :active="current_step" finish-status="success" simple>
        <el-step title="Home" />
        <el-step title="New" />
        <el-step title="Optimize" />
        <el-step title="Result" />
      </el-steps>
    </el-header>
    <el-main style="padding: 0px !important; height: calc(100vh - 100px)">
      <div id="step_0" class="div_selected">
        <el-button size="large" type="primary" style="font-family: 'Segoe UI Symbol'" @click="new_qcda"> New
        </el-button>
        <el-button size="large" type="primary" plain style="margin: 0px 10px; font-family: 'Segoe UI Symbol'"
          @click="load_qcda"> LOAD
        </el-button>

      </div>
      <div id="step_1_N" class="div_not_selected"></div>
      <div id="step_1_L" class="div_not_selected">
        <el-upload class="load_qcda" :action="uploadBackend" :multiple="multipleUpload" :show-file-list="showFileList"
          :before-upload="loadQCDA">
          <el-button size="large" type="primary" plain style="margin: 0px 10px; font-family: 'Segoe UI Symbol'"> LOAD
          </el-button>
        </el-upload>
        <el-button size="large" type="primary" plain @click="confirm_loadQCDA" :enabled="LoadConfirmBtnEnable"
          style="margin: 0px 10px; font-family: 'Segoe UI Symbol'"> Confirm </el-button>
      </div>
      <div id="step_2" class="div_not_selected">
        <oVisualizeZone ref="o_visVue" :VisContentIn="o_VisContent">
          <!-- TODO: replace with a one way vue component -->
        </oVisualizeZone>
        <el-button size="large" type="primary" plain @click="run_o_QCDA" 
          style="margin: 0px 10px; font-family: 'Segoe UI Symbol'"> Confirm </el-button>
      </div>
      <div id="step_3" class="div_not_selected">
        <!-- result div -->
      </div>
    </el-main>
  </el-container>
</template>
<style>
.div_selected {
  display: block;
}

.div_not_selected {
  display: none;
}
</style>
<script>
import * as d3 from "d3";
import oVisualizeZone from "./oVisualizeZone.vue";

export default {
  props: {},
  data: function () {
    return {
      uploadBackend: `${this.background}/api/uploadFile?uid=${this.uuid}`,
      showFileList: false,
      multipleUpload: false,
      current_step: 0,
      ProgramText: "",
      o_VisContent: {
        gateSet: [
        ],
        q: [0, 1, 2, 3, 4],
        gates: [

        ],
      },
      LoadConfirmBtnEnable: false,
    };
  },
  components: {
    oVisualizeZone,
  },
  methods: {
    new_qcda() {
      this.current_step = 1;
      d3.select("#step_0").attr("class", "div_not_selected");
      d3.select("#step_1_N").attr("class", "div_selected");
      d3.select("#step_1_L").attr("class", "div_not_selected");
      d3.select("#step_2").attr("class", "div_not_selected");
      d3.select("#step_3").attr("class", "div_not_selected");
    },
    load_qcda() {
      this.current_step = 1;
      d3.select("#step_0").attr("class", "div_not_selected");
      d3.select("#step_1_N").attr("class", "div_not_selected");
      d3.select("#step_1_L").attr("class", "div_selected");
      d3.select("#step_2").attr("class", "div_not_selected");
      d3.select("#step_3").attr("class", "div_not_selected");
    },
    show_o_qasm() {
      this.current_step = 2;
      d3.select("#step_0").attr("class", "div_not_selected");
      d3.select("#step_1_N").attr("class", "div_not_selected");
      d3.select("#step_1_L").attr("class", "div_not_selected");
      d3.select("#step_2").attr("class", "div_selected");
      d3.select("#step_3").attr("class", "div_not_selected");
    },
    confirm_o_QCDA(){
      this.current_step = 3;
      d3.select("#step_0").attr("class", "div_not_selected");
      d3.select("#step_1_N").attr("class", "div_not_selected");
      d3.select("#step_1_L").attr("class", "div_not_selected");
      d3.select("#step_2").attr("class", "div_not_selected");
      d3.select("#step_3").attr("class", "div_selected");
    },
    loadQCDA(file) {
      // 加载qasm文件
      console.log(file);
      let reader = new FileReader();
      reader.readAsText(file, "UTF-8");

      reader.onload = (evt) => {
        let text = evt.target.result;
        console.log(text);
        this.ProgramText = text;
        this.LoadConfirmBtnEnable = true;

      };

      reader.onerror = (evt) => {
        console.error(evt);
      };
    },
    confirm_loadQCDA() {
      this.socket.emit("qasm_load", {
        uuid: this.uuid,
        content: this.ProgramText,
        source:'QCDA'
      });
    },
    run_o_QCDA(){
      this.socket.emit("o_qasm_run", {
        uuid: this.uuid,
        content: this.ProgramText,
        source:'QCDA'
      });
    },
    append2Group(groupGates, posX, gate) {
      // 将当前gate加入到列表末尾
      console.log("groupGates before", groupGates);
      while (posX >= groupGates.length) {
        groupGates.push([]);
      }
      let group = groupGates[posX];
      for (let i = 0; i < group.length; i++) {
        if (this.checkConflict(group[i], gate)) {
          let groupCopy = JSON.parse(JSON.stringify(group));
          let cut1 = groupCopy;
          let cut2 = [gate];
          groupGates.splice(posX, 1, cut1, cut2);
          return;
        }
      }
      group.push(gate);
      console.log("groupGates after", groupGates);
    },
    checkConflict(gate, joinGate) {
      // 检查当前gate是否可以插入当前x坐标
      let conflicted = false;
      let min = gate.q;
      let max = 0;
      gate.controls.forEach((ctl) => {
        if (ctl > max) {
          max = ctl;
        }
      });
      gate.targets.forEach((tar) => {
        if (tar > max) {
          max = tar;
        }
      });
      joinGate.controls.forEach((ctl) => {
        if (ctl >= min && ctl <= max) {
          conflicted = true;
        }
      });
      joinGate.targets.forEach((tar) => {
        if (tar >= min && tar <= max) {
          conflicted = true;
        }
      });
      return conflicted;
    },
    ListGates(groupGates) {
      // 把gate组转换成1维数组
      let listGates = [];
      let index = 0;
      for (let i = 0; i < groupGates.length; i++) {
        let group = groupGates[i].sort((a, b) => {
          return a.q - b.q;
        });
        for (let j = 0; j < group.length; j++) {
          group[j].posX = i;
          group[j].index = index;
          index += 1;
          listGates.push(group[j]);
        }
      }
      return listGates;
    },
  },
  mounted: function () {
    this.socket.on("QCDA_o_qasm_load", (content) => {
      // 收到后端处理好的o_qasm
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.o_ProgramText = content.qasm;
    });

    this.socket.on("QCDA_o_gates_update", (content) => {
      // 收到后端qasm对应gate列表，在前端显示
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      let groupedGates = [];

      content["gates"].forEach((gate_org) => {
        let gate = {
          q: this.o_VisContent.q.length - 1,
          name: gate_org.name,
          targets: gate_org.targets,
          controls: gate_org.controls,
          selected: false,
          pargs: gate_org.pargs,
          img: gate_org.img,
          qasm_name: gate_org.qasm_name,
        };
        gate_org.controls.forEach((q) => {
          if (q < gate.q) {
            gate.q = q;
          }
        });
        gate_org.targets.forEach((q) => {
          if (q < gate.q) {
            gate.q = q;
          }
        });
        this.append2Group(
          groupedGates,
          groupedGates.length > 0
            ? groupedGates.length - 1
            : groupedGates.length,
          gate
        );
      });
      this.o_VisContent.gates = this.ListGates(groupedGates);

      this.$refs.o_visVue.vis_change();
      this.show_o_qasm()
    });
  },
  watch: {},
  emits: {
    NewQCDA: null,
    LoadQCDA: null,
  },
};
</script>