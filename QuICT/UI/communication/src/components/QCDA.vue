<template>
  <el-container
    style="
      height: calc(100vh - 50px);
      font-size: var(--el-font-size-large);
      width: 100%;
    "
  >
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
        <el-button
          size="large"
          type="primary"
          plain
          style="
            font-family: 'Segoe UI Symbol';
            width: 100px;
            height: 100px;
            margin: 100px 10px;
          "
          @click="new_qcda"
          > New
        </el-button>
        <el-button
          size="large"
          type="primary"
          plain
          style="
            margin: 100px 10px;
            font-family: 'Segoe UI Symbol';
            width: 100px;
            height: 100px;
          "
          @click="load_qcda"
          > LOAD
        </el-button>
      </div>
      <div id="step_1_N" class="div_not_selected">
        <el-container>
          <el-main class="vis-block">
            <ToolBar
              ref="n_toolBar"
              v-on:SaveQCDA="toolbar_func"
              v-on:RunQCDA="toolbar_func"
              v-on:LoadQCDA="toolbar_func"
              v-on:ChangeSet="n_ChangeSet"
              v-on:UpdateCustomerSet="n_UpdateCustomerSet"
              v-on:UpdataTopology="n_UpdataTopology"
              :all_sets="n_all_sets"
              :customer_set="n_customer_set"
              :topology="n_topology"
              :q="n_qbit"
              :id_base="'QCDA_new'"
              :show_save_run_load="false"
            >
            </ToolBar>
            <nVisualizeZone
              ref="n_visVue"
              :VisContentIn="n_VisContent"
              v-on:VisUpdate="n_VisUpdate"
            >
            </nVisualizeZone>
            <el-button
              size="large"
              type="primary"
              plain
              @click="back_qcda"
              style="margin: 0px 10px; font-family: 'Segoe UI Symbol'"
            >
              Back
            </el-button>
            <el-button
              size="large"
              type="primary"
              plain
              @click="confirm_newQCDA"
              :enabled="NewConfirmBtnEnable"
              style="margin: 0px 10px; font-family: 'Segoe UI Symbol'"
            >
              Next
            </el-button>
          </el-main>
          <el-aside width="20%" style="background-color: #292c3d; padding: 0px">
            <ProgramZone
              :ProgramTextIn="n_ProgramText"
              v-on:ProgramUpdate="n_ProgramUpdate"
            >
            </ProgramZone>
          </el-aside>
        </el-container>
      </div>
      <div id="step_1_L" class="div_not_selected">
        <ToolBar
          ref="l_toolBar"
          v-on:SaveQCDA="toolbar_func"
          v-on:RunQCDA="toolbar_func"
          v-on:LoadQCDA="toolbar_func"
          v-on:ChangeSet="l_ChangeSet"
          v-on:UpdateCustomerSet="l_UpdateCustomerSet"
          v-on:UpdataTopology="l_UpdataTopology"
          :all_sets="l_all_sets"
          :customer_set="l_customer_set"
          :topology="l_topology"
          :q="l_qbit"
          :id_base="'QCDA_load'"
          :show_save_run_load="false"
        >
        </ToolBar>
        <el-upload
          class="load_qcda"
          :action="uploadBackend"
          :multiple="multipleUpload"
          :show-file-list="showFileList"
          :before-upload="loadQCDA"
        >
          <el-button
            size="large"
            type="primary"
            plain
            style="
              margin: 100px 10px;
              font-family: 'Segoe UI Symbol';
              width: 100px;
              height: 100px;
            "
            > LOAD
          </el-button>
        </el-upload>
        <el-button
          size="large"
          type="primary"
          plain
          @click="back_qcda"
          style="margin: 0px 10px; font-family: 'Segoe UI Symbol'"
        >
          Back
        </el-button>
        <el-button
          size="large"
          type="primary"
          plain
          @click="confirm_loadQCDA"
          :disabled="LoadConfirmBtnDisable"
          style="margin: 0px 10px; font-family: 'Segoe UI Symbol'"
        >
          Next
        </el-button>
      </div>
      <div id="step_2" class="div_not_selected">
        <oVisualizeZone ref="o_visVue" :VisContentIn="o_VisContent">
          <!-- TODO: replace with a one way vue component -->
        </oVisualizeZone>
        <el-button
          size="large"
          type="primary"
          plain
          @click="back_o_qasm"
          style="margin: 0px 10px; font-family: 'Segoe UI Symbol'"
        >
          Back
        </el-button>
        <el-button
          size="large"
          type="primary"
          plain
          @click="run_o_QCDA"
          style="margin: 0px 10px; font-family: 'Segoe UI Symbol'"
        >
          Next
        </el-button>
      </div>
      <div id="step_3" class="div_not_selected">
        <el-radio-group v-model="Output_type" @change="DrawOutput">
            <el-radio :label="0">Counts</el-radio>
            <el-radio :label="1">State Vector</el-radio>
            <el-radio :label="2">Density Matrix</el-radio>
          </el-radio-group>
        <el-tabs
          type="border-card"
          style="background: transparent !important; border: 0px solid"
          v-if="Output_type == 0"
        >
          <el-tab-pane label="Table">
            <el-row
              style="height: 40px"
              v-if="Object.keys(OutputContent).length > 0"
            >
              <el-col :span="4"></el-col>
              <el-col :span="6"><b>State</b></el-col>
              <el-col :span="4"></el-col>
              <el-col :span="6"><b>Measured</b></el-col>
              <el-col :span="4"></el-col>
            </el-row>

            <el-row
              style="height: 40px"
              v-for="[k, v] in Object.entries(OutputContent).sort()"
              :key="k"
            >
              <el-col :span="4"></el-col>
              <el-col :span="6">{{ k }}</el-col>
              <el-col :span="4"></el-col>
              <el-col :span="6">{{ v }}</el-col>
              <!-- <el-col :span="6" v-if="result[2].startsWith('-')"
                    >{{ result[1]
                    }}{{ result[2].replace("-", " - ") }} j</el-col
                  >
                  <el-col :span="6" v-else
                    >{{ result[1] }} + {{ result[2] }} j</el-col
                  > -->
              <el-col :span="4"></el-col>
            </el-row>
          </el-tab-pane>
          <el-tab-pane label="Histogram">
            <div id="o_histogram"></div>
          </el-tab-pane>
        </el-tabs>
        <el-tabs
          type="border-card"
          style="background: transparent !important; border: 0px solid"
          v-if="Output_type == 1"
        >
          <el-tab-pane label="Table">
            <el-row
              style="height: 40px"
              v-if="OutputContent_state_vector.length > 0"
            >
              <el-col :span="4"></el-col>
              <el-col :span="6"><b>State</b></el-col>
              <el-col :span="4"></el-col>
              <el-col :span="6"><b>Measured</b></el-col>
              <el-col :span="4"></el-col>
            </el-row>

            <el-row
              style="height: 40px"
              v-for="result in OutputContent_state_vector"
              :key="result"
            >
              <el-col :span="4"></el-col>
              <el-col :span="6">{{ result[0] }}</el-col>
              <el-col :span="4"></el-col>
              <el-col :span="6" v-if="result[2].startsWith('-')"
                >{{ result[1] }}{{ result[2].replace("-", " - ") }} j</el-col
              >
              <el-col :span="6" v-else
                >{{ result[1] }} + {{ result[2] }} j</el-col
              >
              <el-col :span="4"></el-col>
            </el-row>
          </el-tab-pane>
          <el-tab-pane label="Histogram">
            <div id="o_histogram_state_vector"></div>
          </el-tab-pane>
        </el-tabs>
        <el-button
          size="large"
          type="primary"
          plain
          @click="back_r_QCDA"
          style="margin: 0px 10px; font-family: 'Segoe UI Symbol'"
        >
          Back
        </el-button>
        <el-button
          size="large"
          type="primary"
          @click="back_qcda"
          style="margin: 0px 10px; font-family: 'Segoe UI Symbol'"
        >
          Restart
        </el-button>
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
import ToolBar from "./ToolBar.vue";

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
        gateSet: [],
        q: [0, 1, 2, 3, 4],
        gates: [],
      },
      n_VisContent: {
        gateSet: [],
        q: [0, 1, 2, 3, 4],
        gates: [],
      },
      o_VisContent: {
        gateSet: [],
        q: [0, 1, 2, 3, 4],
        gates: [],
      },
      LoadConfirmBtnDisable: true,
      NewConfirmBtnEnable: false,
      OutputContent: {},
      OutputContent_state_vector: {},
      Output_type: 0,
      Route: "N",
      n_all_sets: [],
      l_all_sets: [],
      n_customer_set: [],
      l_customer_set: [],
      n_current_set: 0,
      l_current_set: 0,
      n_topology: [],
      l_topology: [],
      n_qbit: [],
      l_qbit: [],
    };
  },
  components: {
    oVisualizeZone,
    nVisualizeZone,
    ProgramZone,
    ToolBar,
  },
  methods: {
    toolbar_func() {
      return true;
    },
    new_qcda() {
      this.current_step = 1;
      this.Route = "N";
      d3.select("#step_0").attr("class", "div_not_selected");
      d3.select("#step_1_N").attr("class", "div_selected");
      d3.select("#step_1_L").attr("class", "div_not_selected");
      d3.select("#step_2").attr("class", "div_not_selected");
      d3.select("#step_3").attr("class", "div_not_selected");
      this.socket.emit("get_gate_set", { uuid: this.uuid, source: "QCDA" });
    },
    load_qcda() {
      this.current_step = 1;
      this.Route = "L";
      d3.select("#step_0").attr("class", "div_not_selected");
      d3.select("#step_1_N").attr("class", "div_not_selected");
      d3.select("#step_1_L").attr("class", "div_selected");
      d3.select("#step_2").attr("class", "div_not_selected");
      d3.select("#step_3").attr("class", "div_not_selected");
      this.socket.emit("get_gate_set", {
        uuid: this.uuid,
        source: "QCDA_load",
      });
    },
    back_qcda() {
      this.current_step = 0;
      d3.select("#step_0").attr("class", "div_selected");
      d3.select("#step_1_N").attr("class", "div_not_selected");
      d3.select("#step_1_L").attr("class", "div_not_selected");
      d3.select("#step_2").attr("class", "div_not_selected");
      d3.select("#step_3").attr("class", "div_not_selected");
      this.n_VisContent = {
        gateSet: [],
        q: [0, 1, 2, 3, 4],
        gates: [],
      };
      this.n_ProgramText = "";
      this.LoadConfirmBtnDisable = true;
    },
    show_o_qasm() {
      this.current_step = 2;
      d3.select("#step_0").attr("class", "div_not_selected");
      d3.select("#step_1_N").attr("class", "div_not_selected");
      d3.select("#step_1_L").attr("class", "div_not_selected");
      d3.select("#step_2").attr("class", "div_selected");
      d3.select("#step_3").attr("class", "div_not_selected");
    },
    back_o_qasm() {
      this.current_step = 1;
      d3.select("#step_0").attr("class", "div_not_selected");
      if (this.Route == "N") {
        d3.select("#step_1_N").attr("class", "div_selected");
        d3.select("#step_1_L").attr("class", "div_not_selected");
      } else if (this.Route == "L") {
        d3.select("#step_1_N").attr("class", "div_not_selected");
        d3.select("#step_1_L").attr("class", "div_selected");
        this.LoadConfirmBtnDisable = true;
      }
      d3.select("#step_2").attr("class", "div_not_selected");
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
    back_r_QCDA() {
      this.current_step = 2;
      d3.select("#step_0").attr("class", "div_not_selected");
      d3.select("#step_1_N").attr("class", "div_not_selected");
      d3.select("#step_1_L").attr("class", "div_not_selected");
      d3.select("#step_2").attr("class", "div_selected");
      d3.select("#step_3").attr("class", "div_not_selected");
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
        this.LoadConfirmBtnDisable = false;
      };

      reader.onerror = (evt) => {
        console.error(evt);
      };
    },
    confirm_loadQCDA() {
      this.socket.emit("qasm_load", {
        uuid: this.uuid,
        content: this.l_ProgramText,
        source: "QCDA",
        optimize: this.$refs.l_toolBar.getOpSwitch(),
        mapping: this.$refs.l_toolBar.getMapSwitch(),
        topology: this.l_topology,
        set: this.l_all_sets[this.l_current_set],
      });
    },
    confirm_newQCDA() {
      this.socket.emit("qasm_load", {
        uuid: this.uuid,
        content: this.n_ProgramText,
        source: "QCDA",
        optimize: this.$refs.n_toolBar.getOpSwitch(),
        mapping: this.$refs.n_toolBar.getMapSwitch(),
        topology: this.n_topology,
        set: this.n_all_sets[this.n_current_set],
      });
    },
    run_o_QCDA() {
      let setting = {};
      if (this.Route == "N") {
        setting = this.$refs.n_toolBar.getSetting();
      } else {
        setting = this.$refs.l_toolBar.getSetting();
      }

      let ProgramText = this.n_ProgramText;
      let optimize = this.$refs.n_toolBar.getOpSwitch();
      let mapping = this.$refs.n_toolBar.getMapSwitch();
      let topology = this.n_topology;
      let set = this.n_all_sets[this.n_current_set];

      if (this.Route == "L") {
        ProgramText = this.l_ProgramText;
        optimize = this.$refs.l_toolBar.getOpSwitch();
        mapping = this.$refs.l_toolBar.getMapSwitch();
        topology = this.l_topology;
        set = this.l_all_sets[this.l_current_set];
      }
      this.socket.emit("o_qasm_run", {
        uuid: this.uuid,
        content: ProgramText,
        source: "QCDA",
        optimize: optimize,
        mapping: mapping,
        topology: topology,
        set: set,
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
        this.n_Q_Add();
        this.$refs.n_visVue.vis_change();
      }
      if (VisAction.type == "q remove") {
        // 删除qbit
        this.n_Q_Remove(VisAction.index);
        this.$refs.n_visVue.vis_change();
      }
      this.n_qbit = this.n_VisContent.q;
      this.n_ProgramText = this.n_GenQASM();
      this.NewConfirmBtnEnable = true;
    },
    n_ProgramUpdate(ProgramText) {
      // 通知后端qasm更新
      this.socket.emit("programe_update", {
        uuid: this.uuid,
        content: ProgramText,
        source: "QCDA",
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
    n_GenQASM() {
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
    l_GenQASM() {
      // 从gate列表生成qasm
      let qasm_string = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n';
      let cbits = 0;
      this.l_VisContent.gates.forEach((gate) => {
        if (gate.name == "MeasureGate") {
          cbits += 1;
        }
      });

      qasm_string += `qreg q[${this.l_VisContent.q.length}];\n`;
      if (cbits != 0) {
        qasm_string += `creg c[${cbits}];\n`;
      }
      cbits = 0;
      this.l_VisContent.gates.forEach((gate) => {
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
    n_Q_Add() {
      // 新加qbit
      this.n_VisContent.q.push(this.n_VisContent.q.length);
    },
    n_Q_Remove(idx) {
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
    l_Q_Add() {
      // 新加qbit
      this.l_VisContent.q.push(this.l_VisContent.q.length);
    },
    l_Q_Remove(idx) {
      // 删除qbit
      for (let i = this.l_VisContent.gates.length - 1; i >= 0; i--) {
        let touched_q = false;
        if (this.l_VisContent.gates[i].q == idx) {
          touched_q = true;
        } else {
          this.l_VisContent.gates[i].controls.forEach((c) => {
            if (c == idx) {
              touched_q = true;
            }
          });
          if (!touched_q) {
            this.l_VisContent.gates[i].targets.forEach((t) => {
              if (t == idx) {
                touched_q = true;
              }
            });
          }
        }
        if (touched_q) {
          this.l_VisContent.gates.splice(i, 1);
        } else if (this.l_VisContent.gates[i].q > idx) {
          this.l_VisContent.gates[i].q--;
          for (let j = 0; j < this.l_VisContent.gates[i].controls.length; j++) {
            this.l_VisContent.gates[i].controls[j] -= 1;
          }
          for (let j = 0; j < this.l_VisContent.gates[i].targets.length; j++) {
            this.l_VisContent.gates[i].targets[j] -= 1;
          }
        }
      }
      for (let i = 0; i < this.l_VisContent.gates.length; i++) {
        this.l_VisContent.gates[i].index = i;
      }
      this.l_VisContent.q.pop();
    },
    DrawHistogram(result) {
      // 绘制Amplitude图
      console.log("DrawHistogram", result);

      let width = Object.entries(result).length * 30 + 100;
      let height = 350;
      let histogram_zone = d3.select("#o_histogram");
      histogram_zone.selectAll("*").remove();
      let chart = this.BarChart(Object.entries(result).sort(), {
        x: (d) => d[0],
        y: (d) => d[1],
        title: (d) => {
          // return `Amplitude:${d3.format(".3f")(d[3])}\nPhase angle:${d[4]}`;
          return `${d[0]}\nCounts:${d[1]}`;
        },
        xDomain: d3.map(Object.entries(result).sort(), (d) => d[0]), // sort by descending frequency
        yFormat: ".1f", //".3f", //"d",
        // yLabel: "Amplitude",
        yLabel: "nCounts",
        width: width,
        height: height,
        color: "steelblue",
      });
      histogram_zone.node().appendChild(chart);
    },
    DrawHistogram_state_vector(result) {
      console.log("DrawHistogram", result);
      let width = result.length * 30 + 100;
      let height = 350;
      let histogram_zone = d3.select("#o_histogram_state_vector");
      histogram_zone.selectAll("*").remove();
      let chart = this.BarChart(result, {
        x: (d) => d[0],
        y: (d) => d[3],
        title: (d) => {
          return `Amplitude:${d3.format(".3f")(d[3])}\nPhase angle:${d[4]}`;
        },
        xDomain: d3.map(result, (d) => d[0]), // sort by descending frequency
        yFormat: ".3f",
        yLabel: "Amplitude",
        width: width,
        height: height,
        color: "steelblue",
      });
      histogram_zone.node().appendChild(chart);
    },
    BarChart( // 用d3绘制barchart
      data,
      {
        x = (d, i) => i, // given d in data, returns the (ordinal) x-value
        y = (d) => d, // given d in data, returns the (quantitative) y-value
        title, // given d in data, returns the title text
        marginTop = 20, // the top margin, in pixels
        marginRight = 0, // the right margin, in pixels
        marginBottom = 30, // the bottom margin, in pixels
        marginLeft = 40, // the left margin, in pixels
        width = 640, // the outer width of the chart, in pixels
        height = 400, // the outer height of the chart, in pixels
        xDomain, // an array of (ordinal) x-values
        xRange = [marginLeft, width - marginRight], // [left, right]
        yType = d3.scaleLinear, // y-scale type
        yDomain, // [ymin, ymax]
        yRange = [height - marginBottom, marginTop], // [bottom, top]
        xPadding = 0.1, // amount of x-range to reserve to separate bars
        yFormat, // a format specifier string for the y-axis
        yLabel, // a label for the y-axis
        color = "currentColor", // bar fill color
      } = {}
    ) {
      // Compute values.
      const X = d3.map(data, x);
      const Y = d3.map(data, y);

      // Compute default domains, and unique the x-domain.
      if (xDomain === undefined) xDomain = X;
      if (yDomain === undefined) yDomain = [0, d3.max(Y)];
      xDomain = new d3.InternSet(xDomain);

      // Omit any data not present in the x-domain.
      const I = d3.range(X.length).filter((i) => xDomain.has(X[i]));

      // Construct scales, axes, and formats.
      const xScale = d3.scaleBand(xDomain, xRange).padding(xPadding);
      const yScale = yType(yDomain, yRange);
      const xAxis = d3.axisBottom(xScale).tickSizeOuter(0);
      const yAxis = d3.axisLeft(yScale).ticks(height / 40, yFormat);

      // Compute titles.
      if (title === undefined) {
        const formatValue = yScale.tickFormat(100, yFormat);
        title = (i) => `${X[i]}\n${formatValue(Y[i])}`;
      } else {
        const O = d3.map(data, (d) => d);
        const T = title;
        title = (i) => T(O[i], i, data);
      }

      const svg = d3
        .create("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .attr("style", "max-width: 100%; height: auto; height: intrinsic;");

      svg
        .append("g")
        .attr("transform", `translate(${marginLeft},0)`)
        .call(yAxis)
        .call((g) => g.select(".domain").remove())
        .call((g) =>
          g
            .selectAll(".tick line")
            .clone()
            .attr("x2", width - marginLeft - marginRight)
            .attr("stroke-opacity", 0.1)
        )
        .call((g) =>
          g
            .append("text")
            .attr("x", -marginLeft)
            .attr("y", 10)
            .attr("fill", "currentColor")
            .attr("text-anchor", "start")
            .text(yLabel)
        );

      const bar = svg
        .append("g")
        .attr("fill", color)
        .selectAll("rect")
        .data(I)
        .join("rect")
        .attr("x", (i) => xScale(X[i]))
        .attr("y", (i) => yScale(Y[i]))
        .attr("height", (i) => yScale(0) - yScale(Y[i]))
        .attr("width", xScale.bandwidth());

      if (title) bar.append("title").text(title);

      svg
        .append("g")
        .attr("transform", `translate(0,${height - marginBottom})`)
        .call(xAxis);

      return svg.node();
    },
    n_ChangeSet(newSet) {
      // 切换instructionSet
      console.log(`set changed: ${newSet}`);
      this.n_current_set = newSet;
      this.n_VisContent.gateSet = this.n_all_sets[this.n_current_set]["gates"];
      this.$refs.n_visVue.vis_change();
    },
    n_UpdateCustomerSet(customerSet) {
      // 更新customerSet
      console.log(`customer Set changed: ${customerSet}`);
      this.n_customer_set = customerSet;
      this.n_all_sets[1].gates = customerSet;
    },
    n_UpdataTopology(topology, qbit) {
      // 更新topology
      this.n_topology = topology;
      while (this.n_VisContent.q.length < qbit.length) {
        this.n_Q_Add();
      }
      while (this.n_VisContent.q.length > qbit.length) {
        this.n_Q_Remove(this.n_VisContent.q.length - 1);
      }
      this.n_qbit = this.n_VisContent.q;
      this.$refs.n_visVue.vis_change();
      this.n_ProgramText = this.n_GenQASM();
    },
    l_ChangeSet(newSet) {
      // 切换instructionSet
      console.log(`set changed: ${newSet}`);
      this.l_current_set = newSet;
      this.l_VisContent.gateSet = this.l_all_sets[this.l_current_set]["gates"];
      // this.$refs.l_visVue.vis_change();
    },
    l_UpdateCustomerSet(customerSet) {
      // 更新customerSet
      console.log(`customer Set changed: ${customerSet}`);
      this.l_customer_set = customerSet;
      this.l_all_sets[1].gates = customerSet;
    },
    l_UpdataTopology(topology, qbit) {
      // 更新topology
      this.l_topology = topology;
      while (this.l_VisContent.q.length < qbit.length) {
        this.l_Q_Add();
      }
      while (this.l_VisContent.q.length > qbit.length) {
        this.l_Q_Remove(this.l_VisContent.q.length - 1);
      }
      this.l_qbit = this.l_VisContent.q;
      // this.$refs.l_visVue.vis_change();
      this.l_ProgramText = this.l_GenQASM();
    },
    DrawOutput(Output_type) {
      switch (Output_type) {
        case 0:
          this.DrawHistogram(this.OutputContent);
          break;
        case 1:
          this.DrawHistogram_state_vector(this.OutputContent_state_vector);
          break;
        default:
          break;
      }
    },
  },
  mounted: function () {
    // let this_ref = this;
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
      this.show_o_qasm();
    });

    this.socket.on("n_all_sets", (content) => {
      // 收到后端instruction Set 列表， 更新前端相关显示
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.n_VisContent.gateSet = content.all_sets[0]["gates"];
      this.n_customer_set = [];
      this.n_all_sets = content.all_sets;
      this.$refs.n_visVue.vis_change();
    });

    this.socket.on("l_all_sets", (content) => {
      // 收到后端instruction Set 列表， 更新前端相关显示
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.l_VisContent.gateSet = content.all_sets[0]["gates"];
      this.l_customer_set = [];
      this.l_all_sets = content.all_sets;

      this.l_qbit = this.l_VisContent.q;
      this.l_ProgramText = this.l_GenQASM();
      // this.$refs.n_visVue.vis_change();
    });

    this.socket.on("o_run_result", (content) => {
      // 收到后端运行qasm结果， 在前端展示
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.OutputContent = content.run_result.data.counts;
      this.OutputContent_state_vector = content.run_result.data.state_vector;
      this.DrawOutput(this.Output_type);
      this.confirm_o_QCDA();
    });
  },
  watch: {},
  emits: {
    NewQCDA: null,
    LoadQCDA: null,
  },
};
</script>