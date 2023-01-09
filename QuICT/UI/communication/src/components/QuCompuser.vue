<template>
  <el-container style="background-color: #13141c; padding: 0px">
    <el-header class="status-bar" style="padding: 0px; height: 50px">
      <ToolBar v-on:SaveQCDA="SaveQCDA" v-on:RunQCDA="RunQCDA" v-on:LoadQCDA="LoadQCDA" v-on:ChangeSet="ChangeSet"
        v-on:UpdateCustomerSet="UpdateCustomerSet" v-on:UpdataTopology="UpdataTopology" :all_sets="all_sets"
        :customer_set="customer_set" :topology="topology" :q="qbit" :id_base="'QuCompuser'" :show_save_run_load="true">
      </ToolBar>
    </el-header>
    <el-container>
      <el-container direction="vertical">
        <el-main class="vis-block">
          <VisualizeZone ref="visVue" :VisContentIn="VisContent" v-on:VisUpdate="VisUpdate">
          </VisualizeZone>
        </el-main>
        <el-main class="status-bar" style="color: #9aa0be; padding: 0px">
          <el-row>
            <el-col :span="4"></el-col>
            <el-col :span="16">{{ StatusContent }}</el-col>
            <el-col :span="4">
              <el-button v-if="ExpandResult" size="small" type="primary" @click="ResultSmall"
                style="font-family: 'Segoe UI Symbol'"></el-button>
              <el-button v-else size="small" type="primary" @click="ResultLarge"
                style="font-family: 'Segoe UI Symbol'"></el-button>
            </el-col>
          </el-row>
        </el-main>
        <el-main class="output-block" style="padding: 0px">
          <el-radio-group v-model="Output_type" @change="DrawOutput">
            <el-radio :label="0">Counts</el-radio>
            <el-radio :label="1">State Vector</el-radio>
            <el-radio :label="2">Density Matrix</el-radio>
          </el-radio-group>

          <el-tabs type="border-card" style="background: transparent !important; border: 0px solid"
            v-if="Output_type == 0">
            <el-tab-pane label="Table">
              <el-row style="height: 40px" v-if="Object.keys(OutputContent).length > 0">
                <el-col :span="4"></el-col>
                <el-col :span="6"><b>State</b></el-col>
                <el-col :span="4"></el-col>
                <el-col :span="6"><b>Measured</b></el-col>
                <el-col :span="4"></el-col>
              </el-row>

              <el-row style="height: 40px" v-for="[k, v] in Object.entries(OutputContent).sort()" :key="k">
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
              <div id="histogram"></div>
            </el-tab-pane>
          </el-tabs>
          <el-tabs type="border-card" style="background: transparent !important; border: 0px solid"
            v-if="Output_type == 1">
            <el-tab-pane label="Table">
              <el-row style="height: 40px" v-if="OutputContent_state_vector.length > 0">
                <el-col :span="4"></el-col>
                <el-col :span="6"><b>State</b></el-col>
                <el-col :span="4"></el-col>
                <el-col :span="6"><b>Measured</b></el-col>
                <el-col :span="4"></el-col>
              </el-row>

              <el-row style="height: 40px" v-for="result in OutputContent_state_vector" :key="result">
                <el-col :span="4"></el-col>
                <el-col :span="6">{{ result[0] }}</el-col>
                <el-col :span="4"></el-col>
                <el-col :span="6" v-if="result[2].startsWith('-')">{{ result[1] }}{{ result[2].replace("-", " - ") }}
                  j</el-col>
                <el-col :span="6" v-else>{{ result[1] }} + {{ result[2] }} j</el-col>
                <el-col :span="4"></el-col>
              </el-row>
            </el-tab-pane>
            <el-tab-pane label="Histogram">
              <div id="histogram_state_vector"></div>
            </el-tab-pane>
          </el-tabs>
          <el-tabs type="border-card" style="background: transparent !important; border: 0px solid"
            v-if="(Output_type == 2)">
            <el-tab-pane label="Table">
              <table style="width: 100%;">
                <tr style="height: 40px" v-for="result in OutputContent_density_matrix" :key="result">
                  <td style="height: 40px" v-for="elem in result" :key="elem">{{elem}}</td>
                </tr>
              </table>
            </el-tab-pane>
          </el-tabs>
        </el-main>
      </el-container>
      <el-aside width="20%" style="background-color: #292c3d; padding: 0px">
        <ProgramZone :ProgramTextIn="ProgramText" v-on:ProgramUpdate="ProgramUpdate">
        </ProgramZone>
      </el-aside>
    </el-container>
  </el-container>
</template>
<style>
.status-bar {
  background-color: #212330;
  color: #333;
  text-align: center;
  height: 50px;
}

.el-row {
  line-height: 50px;
}

.output-block {
  color: #ffffff;
  text-align: center;
  height: 20vh;
}

.el-aside {
  background-color: #d3dce6;
  color: #333;
  text-align: center;
}

.vis-block {
  text-align: center;
  height: calc(100vh - 20vh - 150px);
}

.el-tabs--border-card>.el-tabs__header .el-tabs__item.is-active {
  background-color: transparent !important;
}

.el-tabs--border-card>.el-tabs__header {
  background-color: transparent !important;
}
</style>

<script>
import VisualizeZone from "./VisualizeZone.vue";
import ProgramZone from "./ProgramZone.vue";
import ToolBar from "./ToolBar.vue";
import * as d3 from "d3";

export default {
  props: {},
  data: function () {
    return {
      ProgramText: 'OPENQASM 2.0; \ninclude "qelib1.inc";\nqreg q[5];',
      VisContent: {
        gateSet: [],
        q: [0, 1, 2, 3, 4],
        gates: [],
      },
      all_sets: [],
      current_set: 0,
      customer_set: [],
      topology: [],
      qbit: [0, 1, 2, 3, 4],
      OutputContent: {},
      OutputContent_state_vector: {},
      OutputContent_density_matrix: {},
      Output_type: 0,
      StatusContent: "Create a circuit and run.",
      ExpandResult: false,
    };
  },
  components: {
    VisualizeZone,
    ProgramZone,
    ToolBar,
  },
  methods: {
    ProgramUpdate(ProgramText) {
      // 通知后端qasm更新
      this.socket.emit("programe_update", {
        uuid: this.uuid,
        content: ProgramText,
      });
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
    VisUpdate(VisAction) {
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
        if (posY > this.VisContent.q.length) {
          posY = this.VisContent.q.length - 1;
        }
        if (posY < 0) {
          posY = 0;
        }
        let y_max = posY + VisAction.gate.controls + VisAction.gate.targets - 1;
        if (y_max > this.VisContent.q.length - 1) {
          for (let i = this.VisContent.q.length; i <= y_max; i++) {
            this.VisContent.q.push(i);
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

        let groupedGates = this.GroupGates(this.VisContent.gates);
        this.insert2Group(groupedGates, posX, gate);
        this.VisContent.gates = this.ListGates(groupedGates);

        this.$refs.visVue.vis_change();
      }
      if (VisAction.type == "gates remove") {
        // 删除gate
        this.VisContent.gates.splice(VisAction.index, 1);
        for (let i = 0; i < this.VisContent.gates.length; i++) {
          this.VisContent.gates[i].index = i;
        }
        this.$refs.visVue.vis_change();
      }
      if (VisAction.type == "gates edit") {
        // 编辑gate
        let min = this.VisContent.q.length;
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
        this.VisContent.gates.splice(VisAction.gate.index, 1);
        let groupedGates = this.GroupGates(this.VisContent.gates);
        this.insert2Group(groupedGates, VisAction.gate.posX, VisAction.gate);
        this.VisContent.gates = this.ListGates(groupedGates);

        this.$refs.visVue.vis_change();
      }
      if (VisAction.type == "gates move") {
        // 移动gate
        let posX = VisAction.x;

        if (posX < 0) {
          posX = 0;
        }
        let posY = VisAction.y;
        if (posY > this.VisContent.q.length) {
          posY = this.VisContent.q.length - 1;
        }
        if (posY < 0) {
          posY = 0;
        }

        let min = this.VisContent.q.length - 1;
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
        if (max + delta > this.VisContent.q.length - 1) {
          delta = this.VisContent.q.length - 1 - max;
        }
        VisAction.gate.q += delta;
        for (let i = 0; i < VisAction.gate.controls.length; i++) {
          VisAction.gate.controls[i] += delta;
        }
        for (let i = 0; i < VisAction.gate.targets.length; i++) {
          VisAction.gate.targets[i] += delta;
        }
        console.log(posX, posY, delta, VisAction.gate);

        this.VisContent.gates.splice(VisAction.gate.index, 1);
        let groupedGates = this.GroupGates(this.VisContent.gates);
        this.insert2Group(groupedGates, posX, VisAction.gate);
        this.VisContent.gates = this.ListGates(groupedGates);

        this.$refs.visVue.vis_change();
      }
      if (VisAction.type == "q add") {
        // 新加qbit
        this.Q_Add();
        this.$refs.visVue.vis_change();
      }
      if (VisAction.type == "q remove") {
        // 删除qbit
        this.Q_Remove(VisAction.index);
        this.$refs.visVue.vis_change();
      }
      this.qbit = this.VisContent.q;
      this.ProgramText = this.GenQASM();
    },
    Q_Add() {
      // 新加qbit
      this.VisContent.q.push(this.VisContent.q.length);
    },
    Q_Remove(idx) {
      // 删除qbit
      for (let i = this.VisContent.gates.length - 1; i >= 0; i--) {
        let touched_q = false;
        if (this.VisContent.gates[i].q == idx) {
          touched_q = true;
        } else {
          this.VisContent.gates[i].controls.forEach((c) => {
            if (c == idx) {
              touched_q = true;
            }
          });
          if (!touched_q) {
            this.VisContent.gates[i].targets.forEach((t) => {
              if (t == idx) {
                touched_q = true;
              }
            });
          }
        }
        if (touched_q) {
          this.VisContent.gates.splice(i, 1);
        } else if (this.VisContent.gates[i].q > idx) {
          this.VisContent.gates[i].q--;
          for (let j = 0; j < this.VisContent.gates[i].controls.length; j++) {
            this.VisContent.gates[i].controls[j] -= 1;
          }
          for (let j = 0; j < this.VisContent.gates[i].targets.length; j++) {
            this.VisContent.gates[i].targets[j] -= 1;
          }
        }
      }
      for (let i = 0; i < this.VisContent.gates.length; i++) {
        this.VisContent.gates[i].index = i;
      }
      this.VisContent.q.pop();
    },
    GenQASM() {
      // 从gate列表生成qasm
      let qasm_string = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n';
      let cbits = 0;
      this.VisContent.gates.forEach((gate) => {
        if (gate.name == "MeasureGate") {
          cbits += 1;
        }
      });

      qasm_string += `qreg q[${this.VisContent.q.length}];\n`;
      if (cbits != 0) {
        qasm_string += `creg c[${cbits}];\n`;
      }
      cbits = 0;
      this.VisContent.gates.forEach((gate) => {
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
    SaveQCDA() {
      // 通知后端保存qasm
      this.socket.emit("qasm_save", {
        uuid: this.uuid,
        content: this.ProgramText,
      });
    },
    RunQCDA(opSwitch, mapSwitch, setting) {
      // 通知后端运行qasm
      this.socket.emit("qasm_run", {
        uuid: this.uuid,
        content: this.ProgramText,
        optimize: opSwitch,
        mapping: mapSwitch,
        topology: this.topology,
        set: this.all_sets[this.current_set],
        setting: setting,
      });
    },
    LoadQCDA(file) {
      // 通知后端加载qasm
      console.log(file);
      let reader = new FileReader();
      reader.readAsText(file, "UTF-8");

      reader.onload = (evt) => {
        let text = evt.target.result;
        console.log(text);
        this.socket.emit("qasm_load", {
          uuid: this.uuid,
          content: text,
          source: "QuCompuser",
        });
      };

      reader.onerror = (evt) => {
        console.error(evt);
      };
    },
    ResultSmall() {
      // 切换运行结果显示到小窗口
      this.ExpandResult = false;
      d3.select(".output-block").style("height", "20vh");
      d3.select(".vis-block").style("height", "calc(100vh - 20vh - 150px)");
    },
    ResultLarge() {
      // 切换运行结果显示到大窗口
      this.ExpandResult = true;
      d3.select(".vis-block").style("height", "calc(100vh - 60vh - 150px)");
      d3.select(".output-block").style("height", "60vh");
    },
    DrawHistogram(result) {
      // 绘制Amplitude图
      console.log("DrawHistogram", result);

      let width = Object.entries(result).length * 30 + 100;
      let height = 350;
      let histogram_zone = d3.select("#histogram");
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
      let histogram_zone = d3.select("#histogram_state_vector");
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
    DrawHistogram_density_matrix(result) {
      console.log("DrawHistogram", result);
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
    ChangeSet(newSet) {
      // 切换instructionSet
      console.log(`set changed: ${newSet}`);
      this.current_set = newSet;
      this.VisContent.gateSet = this.all_sets[this.current_set]["gates"];
      this.$refs.visVue.vis_change();
    },
    UpdateCustomerSet(customerSet) {
      // 更新customerSet
      console.log(`customer Set changed: ${customerSet}`);
      this.customer_set = customerSet;
      this.all_sets[1].gates = customerSet;
    },
    UpdataTopology(topology, qbit) {
      // 更新topology
      this.topology = topology;
      while (this.VisContent.q.length < qbit.length) {
        this.Q_Add();
      }
      while (this.VisContent.q.length > qbit.length) {
        this.Q_Remove(this.VisContent.q.length - 1);
      }
      this.qbit = this.VisContent.q;
      this.$refs.visVue.vis_change();
      this.ProgramText = this.GenQASM();
    },
    DrawOutput(Output_type) {
      switch (Output_type) {
        case 0:
          this.DrawHistogram(this.OutputContent);
          break;
        case 1:
          this.DrawHistogram_state_vector(this.OutputContent_state_vector);
          break;
        case 2:
          this.DrawHistogram_density_matrix(this.OutputContent_density_matrix);
          break;
        default:
          break;
      }
    },
  },
  mounted: function () {
    this.socket.on("qasm_load", (content) => {
      // 收到后端处理好的qasm，显示到前端qasm编辑区域
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.ProgramText = content.qasm;
    });

    this.socket.on("download_uri", (content) => {
      // 收到后端qasm文件，在前端开始下载
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      window.location.href += `${this.background}/${content.download_uri}`;
    });

    this.socket.on("run_result", (content) => {
      // 收到后端运行qasm结果， 在前端展示
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.OutputContent = content.run_result.data.counts;
      this.OutputContent_state_vector = content.run_result.data.state_vector;
      this.OutputContent_density_matrix = content.run_result.data.density_matrix;
      this.DrawOutput(this.Output_type);
    });

    this.socket.on("all_sets", (content) => {
      // 收到后端instruction Set 列表， 更新前端相关显示
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.VisContent.gateSet = content.all_sets[0]["gates"];
      this.customer_set = [];
      this.all_sets = content.all_sets;
      this.$refs.visVue.vis_change();
    });
    this.socket.on("info", (content) => {
      // 收到后端信息， 在前端显示
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      this.StatusContent = content.info;
      this.$refs.visVue.vis_change();
    });

    this.socket.on("gates_update", (content) => {
      // 收到后端qasm对应gate列表，在前端显示
      console.log(content);
      if (!content.uuid == this.uuid) {
        return;
      }
      let groupedGates = [];

      content["gates"].forEach((gate_org) => {
        let gate = {
          q: this.VisContent.q.length - 1,
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
      this.VisContent.gates = this.ListGates(groupedGates);

      this.$refs.visVue.vis_change();
    });
    // this.qbit = this.VisContent.q;
  },
  watch: {},
};
</script>