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
      <div id="step_1_N" class="div_not_selected">
        <el-container>
          <el-main class="vis-block">
            <nVisualizeZone ref="n_visVue" :VisContentIn="n_VisContent" v-on:VisUpdate="n_VisUpdate">
            </nVisualizeZone>
          </el-main>
          <el-aside width="20%" style="background-color: #292c3d; padding: 0px">
            <ProgramZone :ProgramTextIn="n_ProgramText" v-on:ProgramUpdate="n_ProgramUpdate">
            </ProgramZone>
          </el-aside>

        </el-container>
        <el-button size="large" type="primary" plain @click="confirm_newQCDA" :enabled="NewConfirmBtnEnable"
          style="margin: 0px 10px; font-family: 'Segoe UI Symbol'"> Confirm </el-button>
      </div>
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
import nVisualizeZone from "./nVisualizeZone.vue";
import ProgramZone from "./ProgramZone.vue";

export default {
  props: {},
  data: function () {
    return {
      uploadBackend: `${this.background}/api/uploadFile?uid=${this.uuid}`,
      showFileList: false,
      multipleUpload: false,
      current_step: 0,
      l_ProgramText: "",
      n_ProgramText: "",
      o_ProgramText: "",
      l_VisContent: {
        gateSet: [
        ],
        q: [0, 1, 2, 3, 4],
        gates: [

        ],
      },
      n_VisContent: {
        gateSet: [
        ],
        q: [0, 1, 2, 3, 4],
        gates: [

        ],
      },
      o_VisContent: {
        gateSet: [
        ],
        q: [0, 1, 2, 3, 4],
        gates: [

        ],
      },
      LoadConfirmBtnEnable: false,
      NewConfirmBtnEnable: false,
    };
  },
  components: {
    oVisualizeZone,
    nVisualizeZone,
    ProgramZone,
  },
  methods: {
    new_qcda() {
      this.current_step = 1;
      d3.select("#step_0").attr("class", "div_not_selected");
      d3.select("#step_1_N").attr("class", "div_selected");
      d3.select("#step_1_L").attr("class", "div_not_selected");
      d3.select("#step_2").attr("class", "div_not_selected");
      d3.select("#step_3").attr("class", "div_not_selected");
      this.socket.emit("get_gate_set", { uuid: this.uuid, source:'QCDA' });
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
    confirm_o_QCDA() {
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
        this.l_ProgramText = text;
        this.LoadConfirmBtnEnable = true;

      };

      reader.onerror = (evt) => {
        console.error(evt);
      };
    },
    confirm_loadQCDA() {
      this.socket.emit("qasm_load", {
        uuid: this.uuid,
        content: this.l_ProgramText,
        source: 'QCDA'
      });
    },
    confirm_newQCDA() {
      this.socket.emit("qasm_load", {
        uuid: this.uuid,
        content: this.n_ProgramText,
        source: 'QCDA'
      });
    },
    run_o_QCDA() {
      let setting = {};
      setting.device = "GPU"; //this.dialogBe;
      setting.shots = 1; //Number(this.dialogSeShots);
      setting.backend = "statevector"; //
      setting.gpu_device_id = 0; //Number(this.dialogSe_GPU_device_id);
      setting.sync = true; //this.dialogSe_sync;
      setting.optimize = true; //this.dialogSe_optimize;
      // switch (setting.device) {
      //   case "CPU":
      //     setting.backend = "unitary";
      //     setting.precision = this.dialogSe_Precision;
      //     break;
      //   case "GPU":
      //     setting.backend = this.dialogBe_Backend;
      //     setting.precision = this.dialogSe_Precision;
      //     switch (this.dialogBe_Backend) {
      //       case "unitary":
      //         break;
      //       case "statevector":
      //         setting.gpu_device_id = Number(this.dialogSe_GPU_device_id);
      //         setting.sync = this.dialogSe_sync;
      //         setting.optimize = this.dialogSe_optimize;
      //         break;
      //       case "multiGPU":
      //         setting.ndev = Number(this.dialogSe_ndev);
      //         setting.sync = this.dialogSe_sync;
      //         break;
      //     }
      //     break;
      //   case "qiskit":
      //     setting.token = Number(this.dialogSeToken);
      //     break;
      //   case "qcompute":
      //     setting.token = Number(this.dialogSeToken);
      //     break;
      // }

      this.socket.emit("o_qasm_run", {
        uuid: this.uuid,
        content: this.o_ProgramText,
        source: 'QCDA',
        optimize: true, //opSwitch,
        mapping: true, //mapSwitch,
        topology: [], //this.topology,
        set: this.n_VisContent.gateSet,
        setting: setting,
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
    n_VisUpdate(VisAction) {
      // 更新gate列表
      // this.ProgramText = VisContent;
      console.log(VisAction);

      if (VisAction.type == "gates add") {
        // 新加gate
        let posX = VisAction.x;

        if (posX < 0) {
          posX = 0;
        }
        let posY = VisAction.y;
        if (posY > this.n_VisContent.q.length) {
          posY = this.n_VisContent.q.length - 1;
        }
        if (posY < 0) {
          posY = 0;
        }
        let y_max = posY + VisAction.gate.controls + VisAction.gate.targets - 1;
        if (y_max > this.n_VisContent.q.length - 1) {
          for (let i = this.n_VisContent.q.length; i <= y_max; i++) {
            this.n_VisContent.q.push(i);
          }
        }
        let gate = {
          q: posY,
          name: VisAction.gate.name,
          targets: [],
          controls: [],
          selected: false,
          pargs: [],
          img: VisAction.gate.img,
          qasm_name: VisAction.gate.qasm_name,
        };
        for (let i = 0; i < VisAction.gate.controls; i++) {
          gate.controls.push(posY + i);
        }
        for (
          let i = VisAction.gate.controls;
          i < VisAction.gate.targets + VisAction.gate.controls;
          i++
        ) {
          gate.targets.push(posY + i);
        }
        VisAction.gate.pargs.forEach((element) => {
          gate.pargs.push(element);
        });

        let groupedGates = this.GroupGates(this.n_VisContent.gates);
        this.insert2Group(groupedGates, posX, gate);
        this.n_VisContent.gates = this.ListGates(groupedGates);

        this.$refs.n_visVue.vis_change();
      }
      if (VisAction.type == "gates remove") {
        // 删除gate
        this.n_VisContent.gates.splice(VisAction.index, 1);
        for (let i = 0; i < this.n_VisContent.gates.length; i++) {
          this.n_VisContent.gates[i].index = i;
        }
        this.$refs.n_visVue.vis_change();
      }
      if (VisAction.type == "gates edit") {
        // 编辑gate
        let min = this.n_VisContent.q.length;
        VisAction.gate.targets.forEach((element) => {
          if (element < min) {
            min = element;
          }
        });
        VisAction.gate.controls.forEach((element) => {
          if (element < min) {
            min = element;
          }
        });
        VisAction.gate.q = min;
        this.n_VisContent.gates.splice(VisAction.gate.index, 1);
        let groupedGates = this.GroupGates(this.n_VisContent.gates);
        this.insert2Group(groupedGates, VisAction.gate.posX, VisAction.gate);
        this.n_VisContent.gates = this.ListGates(groupedGates);

        this.$refs.n_visVue.vis_change();
      }
      if (VisAction.type == "gates move") {
        // 移动gate
        let posX = VisAction.x;

        if (posX < 0) {
          posX = 0;
        }
        let posY = VisAction.y;
        if (posY > this.n_VisContent.q.length) {
          posY = this.n_VisContent.q.length - 1;
        }
        if (posY < 0) {
          posY = 0;
        }

        let min = this.n_VisContent.q.length - 1;
        let max = 0;
        VisAction.gate.controls.forEach((element) => {
          if (element > max) {
            max = element;
          }
          if (element < min) {
            min = element;
          }
        });
        VisAction.gate.targets.forEach((element) => {
          if (element > max) {
            max = element;
          }
          if (element < min) {
            min = element;
          }
        });
        let delta = posY - VisAction.gate.q;
        if (min + delta < 0) {
          delta = 0 - min;
        }
        if (max + delta > this.n_VisContent.q.length - 1) {
          delta = this.n_VisContent.q.length - 1 - max;
        }
        VisAction.gate.q += delta;
        for (let i = 0; i < VisAction.gate.controls.length; i++) {
          VisAction.gate.controls[i] += delta;
        }
        for (let i = 0; i < VisAction.gate.targets.length; i++) {
          VisAction.gate.targets[i] += delta;
        }
        console.log(posX, posY, delta, VisAction.gate);

        this.n_VisContent.gates.splice(VisAction.gate.index, 1);
        let groupedGates = this.GroupGates(this.n_VisContent.gates);
        this.insert2Group(groupedGates, posX, VisAction.gate);
        this.n_VisContent.gates = this.ListGates(groupedGates);

        this.$refs.n_visVue.vis_change();
      }
      if (VisAction.type == "q add") {
        // 新加qbit
        this.Q_Add();
        this.$refs.n_visVue.vis_change();
      }
      if (VisAction.type == "q remove") {
        // 删除qbit
        this.Q_Remove(VisAction.index);
        this.$refs.n_visVue.vis_change();
      }
      this.qbit = this.n_VisContent.q;
      this.n_ProgramText = this.GenQASM();
      this.NewConfirmBtnEnable = true;
    },
    n_ProgramUpdate(ProgramText) {
      // 通知后端qasm更新
      this.socket.emit("programe_update", {
        uuid: this.uuid,
        content: ProgramText,
        source: 'QCDA',
      });
      this.NewConfirmBtnEnable = true;
    },
    GroupGates(Gates) {
      // 把gate按x轴分组
      let groupGates = [];
      Gates.forEach((gate) => {
        while (gate.posX >= groupGates.length) {
          groupGates.push([]);
        }
        groupGates[gate.posX].push(gate);
      });

      return groupGates;
    },
    insert2Group(groupGates, posX, gate) {
      // 插入当前gate到指定x坐标
      console.log("groupGates before", groupGates);
      while (posX >= groupGates.length) {
        groupGates.push([]);
      }
      let group = groupGates[posX];
      for (let i = 0; i < group.length; i++) {
        if (this.checkConflict(group[i], gate)) {
          if (i == 0) {
            let cut1 = [gate];
            groupGates.splice(posX, 0, cut1);
          } else {
            let groupCopy = JSON.parse(JSON.stringify(group));
            let cut1 = groupCopy.splice(0, i);
            cut1.push(gate);
            let cut2 = groupCopy;
            groupGates.splice(posX, 1, cut1, cut2);
          }
          return;
        }
      }
      group.push(gate);
      console.log("groupGates after", groupGates);
    },
    GenQASM() {
      // 从gate列表生成qasm
      let qasm_string = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n';
      let cbits = 0;
      this.n_VisContent.gates.forEach((gate) => {
        if (gate.name == "MeasureGate") {
          cbits += 1;
        }
      });

      qasm_string += `qreg q[${this.n_VisContent.q.length}];\n`;
      if (cbits != 0) {
        qasm_string += `creg c[${cbits}];\n`;
      }
      cbits = 0;
      this.n_VisContent.gates.forEach((gate) => {
        if (gate.name == "MeasureGate") {
          qasm_string += `measure q[${gate.targets[0]}] -> c[${cbits}];\n`;
          cbits += 1;
        } else {
          let qasm = this.qasm(gate);
          if (qasm == "error") {
            console.log(
              "the circuit cannot be transformed to a valid describe in OpenQASM 2.0"
            );
            console.log(gate);
            return "error";
          }
          qasm_string += this.qasm(gate);
        }
      });
      return qasm_string;
    },
    qasm(gate) {
      // 从gate生成qasm片段
      let qasm_string = gate.qasm_name;
      if (gate.pargs.length > 0) {
        qasm_string += "(";
        for (let i = 0; i < gate.pargs.length; i++) {
          if (i != 0) {
            qasm_string += ", ";
          }
          qasm_string += String(gate.pargs[i]);
        }
        qasm_string += ")";
      }
      qasm_string += " ";
      let first_in = true;
      gate.controls.forEach((p) => {
        if (!first_in) {
          qasm_string += ", ";
        } else {
          first_in = false;
        }
        qasm_string += `q[${p}]`;
      });
      gate.targets.forEach((p) => {
        if (!first_in) {
          qasm_string += ", ";
        } else {
          first_in = false;
        }
        qasm_string += `q[${p}]`;
      });
      qasm_string += ";\n";
      return qasm_string;
    },
    Q_Add() {
      // 新加qbit
      this.n_VisContent.q.push(this.n_VisContent.q.length);
    },
    Q_Remove(idx) {
      // 删除qbit
      for (let i = this.n_VisContent.gates.length - 1; i >= 0; i--) {
        let touched_q = false;
        if (this.n_VisContent.gates[i].q == idx) {
          touched_q = true;
        } else {
          this.n_VisContent.gates[i].controls.forEach((c) => {
            if (c == idx) {
              touched_q = true;
            }
          });
          if (!touched_q) {
            this.n_VisContent.gates[i].targets.forEach((t) => {
              if (t == idx) {
                touched_q = true;
              }
            });
          }
        }
        if (touched_q) {
          this.n_VisContent.gates.splice(i, 1);
        } else if (this.n_VisContent.gates[i].q > idx) {
          this.n_VisContent.gates[i].q--;
          for (let j = 0; j < this.n_VisContent.gates[i].controls.length; j++) {
            this.n_VisContent.gates[i].controls[j] -= 1;
          }
          for (let j = 0; j < this.n_VisContent.gates[i].targets.length; j++) {
            this.n_VisContent.gates[i].targets[j] -= 1;
          }
        }
      }
      for (let i = 0; i < this.n_VisContent.gates.length; i++) {
        this.n_VisContent.gates[i].index = i;
      }
      this.n_VisContent.q.pop();
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

    this.socket.on("n_all_sets", (content) => {
      // 收到后端instruction Set 列表， 更新前端相关显示
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.n_VisContent.gateSet = content.all_sets[0]["gates"];
      this.customer_set = [];
      let customer_set = { name: "CustomerSet", gates: [] };
      this.all_sets = content.all_sets;
      this.all_sets.push(customer_set);
      this.$refs.n_visVue.vis_change();
    });

  },
  watch: {},
  emits: {
    NewQCDA: null,
    LoadQCDA: null,
  },
};
</script>