<template>
  <el-row justify="end" style="line-hight: 40px">
    <el-col :span="3">
      <el-button
        size="small"
        type="primary"
        @click="dialogCmdVisible = true"
        style="font-family: 'Segoe UI Symbol'"
        >Instruction Set ⏷</el-button
      >
    </el-col>
    <el-dialog
      title="Instruction Set"
      v-model="dialogCmdVisible"
      width="30%"
      :before-close="handleCmdClose"
    >
      <div style="text-align: left">
        <el-radio
          v-for="instruction in all_sets"
          :key="instruction"
          :label="all_sets.indexOf(instruction)"
          v-model="dialogCmd"
          style="display: inline-flex"
          >{{ instruction.name }}</el-radio
        >
        <div>
          <span style="display: block"
            ><b>Customer Set</b>(Click to remove from Customer Set)</span
          >
          <img
            v-for="gate in customerSet"
            :key="gate"
            :src="'./assets/gate_set/' + gate.img"
            @click="
              () => {
                RemoveFromCustomerSet(gate);
              }
            "
            style="display: inline-flex"
          />
          <span
            v-if="customerSet.length == 0"
            style="display: block; text-align: center"
            >No gate in customer set.</span
          >
        </div>
        <div>
          <span style="display: block">Click to add to Customer Set</span>
          <img
            v-for="gate in tempSet"
            :key="gate"
            :src="'./assets/gate_set/' + gate.img"
            @click="
              () => {
                AddToCustomerSet(gate);
              }
            "
            style="display: inline-flex"
          />
        </div>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogCmdVisible = false">Cancel</el-button>
          <el-button type="primary" @click="dialogCmdVisible = false"
            >OK</el-button
          >
        </span>
      </template>
    </el-dialog>

    <el-col :span="3">
      <el-button size="small" type="primary" @click="showTopologyEdit"
        ><img src="/assets/topology.2x.png" />Topology</el-button
      >
    </el-col>
    <el-dialog
      title="Topology"
      v-model="dialogTpVisible"
      width="30%"
      :before-close="handleTpClose"
    >
      <div>
        <div>
          <el-input
            v-model="dialogTpNodeCount"
            label="n="
            @change="TpNodeCountChange"
          ></el-input>
        </div>
        <div id="topologyZone">
          <svg />
        </div>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-upload
            class="upload-demo"
            :action="uploadBackend"
            :multiple="multipleUpload"
            :show-file-list="showFileList"
            :before-upload="TpLoad"
            style="display: inline-block"
          >
            <el-button
              size="small"
              type="primary"
              style="font-family: 'Segoe UI Symbol'"
              > LOAD</el-button
            >
          </el-upload>
          <el-button @click="pvClear">Clear</el-button>
          <el-button @click="pvAll">All</el-button>
          <el-button @click="pvReverse">Reverse</el-button>
          <el-button
            type="primary"
            @click="
              dialogTpVisible = false;
              TpConfirm();
            "
            >OK</el-button
          >
        </span>
      </template>
    </el-dialog>

    <el-col :span="3">
      <el-checkbox v-model="opSwitch" style="line-hight: 20px"
        >Optimize</el-checkbox
      >

      <el-checkbox v-model="mapSwitch" style="line-hight: 20px"
        >Mapping</el-checkbox
      >
    </el-col>

    <el-col :span="3">
      <el-button
        size="small"
        type="primary"
        @click="dialogBeVisible = true"
        style="font-family: 'Segoe UI Symbol'"
        >Backend ⏷</el-button
      >
    </el-col>
    <el-dialog
      title="Backend"
      v-model="dialogBeVisible"
      width="30%"
      :before-close="handleBeClose"
    >
      <div>
        <span>Device</span>
        <el-radio v-model="dialogBe" label="CPU">CPU</el-radio>
        <el-radio v-model="dialogBe" label="GPU">GPU</el-radio>
        <el-radio v-model="dialogBe" label="qiskit">qiskit</el-radio>
        <el-radio v-model="dialogBe" label="qcompute">qcompute</el-radio>
      </div>
      <div>
        
        <el-select placeholder="Backend" v-if="dialogBe == 'CPU'" v-model="dialogBe_Backend">
          <span>Backend</span>
          <el-option key="unitary" label="unitary" value="unitary">
          </el-option>
        </el-select>
        <el-select
          v-model="dialogBe_Backend"
          placeholder="Backend"
          v-if="dialogBe == 'GPU'"
        >
        <span>Backend</span>
          <el-option
            v-for="item in dialogBe_Backend_options"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          >
          </el-option>
        </el-select>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogBeVisible = false">Cancel</el-button>
          <el-button type="primary" @click="dialogBeVisible = false"
            >OK</el-button
          >
        </span>
      </template>
    </el-dialog>

    <el-col :span="3">
      <el-button
        size="small"
        type="primary"
        @click="dialogSeVisible = true"
        style="font-family: 'Segoe UI Symbol'"
        > Setting</el-button
      >
    </el-col>
    <el-dialog
      title="Setting"
      v-model="dialogSeVisible"
      width="30%"
      :before-close="handleSeClose"
    >
      <div>
         <span style="display: block; text-align: center">Shots</span>
        <el-input v-model="dialogSeShots" label="n="></el-input>
        <div v-if="dialogBe == 'CPU' || dialogBe == 'GPU'">
          <span style="display: block; text-align: center">Precision</span>
          <el-radio v-model="dialogSe_Precision" label="single"
            >single</el-radio
          >
          <el-radio v-model="dialogSe_Precision" label="double"
            >double</el-radio
          >
        </div>
        <div v-if="dialogBe == 'GPU' && dialogBe_Backend == 'statevector'">
          <span style="display: block; text-align: center" id="GPU_device_ID">GPU device ID</span>
          <el-input v-model="dialogSe_GPU_device_id" label="GPU_device_ID"
            >0</el-input
          >
          <el-checkbox v-model="dialogSe_sync" style="line-hight: 20px"
            >Sync</el-checkbox
          >
          <el-checkbox v-model="dialogSe_optimize" style="line-hight: 20px"
            >Optimize</el-checkbox
          >
        </div>
        <div v-if="dialogBe == 'GPU' && dialogBe_Backend == 'multiGPU'">
          <el-input v-model="dialogSe_ndev" label="NDEV"
            >0</el-input
          >
          <el-checkbox v-model="dialogSe_sync" style="line-hight: 20px"
            >Sync</el-checkbox
          >
        </div>
        <div v-if="dialogBe == 'qiskit' || dialogBe == 'qcompute'">
          <span style="display: block; text-align: center" id="Token">Token</span>
          <el-input v-model="dialogSeToken" label="Token"></el-input>
        </div>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogSeVisible = false">Cancel</el-button>
          <el-button type="primary" @click="dialogSeVisible = false"
            >OK</el-button
          >
        </span>
      </template>
    </el-dialog>

    <el-col
      :span="6"
      style="
        display: inline-flex;
        justify-content: flex-end;
        align-items: center;
      "
    >
      <el-upload
        class="upload-demo"
        :action="uploadBackend"
        :multiple="multipleUpload"
        :show-file-list="showFileList"
        :before-upload="loadQCDA"
      >
        <el-button
          size="small"
          type="primary"
          style="font-family: 'Segoe UI Symbol'"
          > LOAD</el-button
        >
      </el-upload>

      <el-button
        size="small"
        type="primary"
        @click="saveQCDA"
        style="font-family: 'Segoe UI Symbol'"
        > SAVE</el-button
      >

      <el-button
        size="small"
        type="primary"
        @click="runQCDA"
        style="margin-right: 15px; font-family: 'Segoe UI Symbol'"
      >
         RUN</el-button
      >
    </el-col>
  </el-row>
</template>
<style scoped>
.upload-demo {
  margin-right: 10px;
}
</style>

<script>
import * as d3 from "d3";
export default {
  props: {
    all_sets: Array,
    customer_set: Array,
    topology: Array,
    q: Array,
  },
  data: function () {
    return {
      tick: 0,
      updated: false,
      InCache: "",
      uploadBackend: `${this.background}/api/uploadFile?uid=${this.uuid}`,
      showFileList: false,
      multipleUpload: false,
      dialogCmdVisible: false,
      dialogTpVisible: false,
      dialogBeVisible: false,
      dialogSeVisible: false,
      dialogCmd: 0,
      dialogBe: `CPU`,
      opSwitch: false,
      mapSwitch: false,
      customerSet: [],
      fullSet: [],
      tempSet: [],
      tp: [],
      fullTp: [],
      qbit: [],
      topologyZone: undefined,
      dialogTpNodeCount: 0,
      dialogSeShots: 1,
      dialogSe_Precision: "single",
      precision_need_show: false,
      dialogBe_Backend: `unitary`,
      dialogBe_Backend_options: [
        { value: `unitary`, label: `unitary` },
        { value: `statevector`, label: `statevector` },
        { value: `multiGPU`, label: `multiGPU` },
      ],
      dialogSe_GPU_device_id:0,
      dialogSe_sync:true,
      dialogSe_optimize:false,
      dialogSe_ndev:1,
      dialogSeToken:1,
    };
  },
  methods: {
    handleCmdClose(done) {
      this.$confirm("确认关闭？")
        .then(() => {
          console.log(this.dialogCmd);
          done();
        })
        .catch(() => {});
    },
    handleTpClose(done) {
      this.$confirm("确认关闭？")
        .then(() => {
          console.log(this.dialogCmd);
          done();
        })
        .catch(() => {});
    },
    handleBeClose(done) {
      this.$confirm("确认关闭？")
        .then(() => {
          console.log(this.dialogCmd);
          this.TpCancel();
          done();
        })
        .catch(() => {});
    },
    showTopologyEdit() {
      this.dialogTpVisible = true;
      this.ResetTopology();
      this.ResetQbit();
      setTimeout(this.updateTopology, 500);
    },
    pvClear() {
      // 清空topology
      this.tp = [];
      this.updateTopology();
    },
    pvAll() {
      // topology设为全连接
      this.pvClear();
      for (let i = 0; i < this.qbit.length; i++) {
        for (let j = i + 1; j < this.qbit.length; j++) {
          this.tp.push(`${this.qbit[i]}_${this.qbit[j]}`);
        }
      }
      this.updateTopology();
    },
    pvReverse() {
      // topology反选
      let t_tp = [];
      for (let i = 0; i < this.qbit.length; i++) {
        for (let j = i + 1; j < this.qbit.length; j++) {
          let uvExist = false;
          for (let k = this.tp.length - 1; k >= 0; k--) {
            if (
              this.tp[k] == `${this.qbit[i]}_${this.qbit[j]}` ||
              this.tp[k] == `${this.qbit[j]}_${this.qbit[i]}`
            ) {
              uvExist = true;
              break;
            }
          }
          if (!uvExist) {
            t_tp.push(`${this.qbit[i]}_${this.qbit[j]}`);
          }
        }
      }
      this.tp = t_tp;
      this.updateTopology();
    },
    pvSwitch(u, v) {
      // 点击连接，在有边和无边间切换
      let uvExist = false;
      for (let i = this.tp.length - 1; i >= 0; i--) {
        if (this.tp[i] == `${u}_${v}` || this.tp[i] == `${v}_${u}`) {
          this.tp.splice(i, 1);
          uvExist = true;
        }
      }
      if (!uvExist) {
        this.tp.push(`${u}_${v}`);
      }
      this.updateTopology();
    },
    updateTopology() {
      // 更新topology图
      //draw control-target zone
      if (this.topologyZone != undefined) {
        d3.select("#topologyZone").select("svg").selectAll("*").remove();
      }
      this.topologyZone = d3
        .select("#topologyZone")
        .select("svg")
        .attr("width", "340px")
        .attr("height", "340px")
        .attr("background", "blue")
        .append("g")
        .style("transform", () => {
          return `translateX(20px) translateY(20px)`;
        });

      //draw connect lines
      this.topologyZone
        .selectAll(".tpFull")
        .data(this.fullTp)
        .join("line")
        .classed("tpFull", true)
        .attr("stroke", "gray")
        .attr("stroke-width", 5)
        .attr("x1", (d) => {
          return (
            150 + 150 * Math.cos((d[0] * 2 * Math.PI) / this.dialogTpNodeCount)
          );
        })
        .attr("y1", (d) => {
          return (
            150 + 150 * Math.sin((d[0] * 2 * Math.PI) / this.dialogTpNodeCount)
          );
        })
        .attr("x2", (d) => {
          return (
            150 + 150 * Math.cos((d[1] * 2 * Math.PI) / this.dialogTpNodeCount)
          );
        })
        .attr("y2", (d) => {
          return (
            150 + 150 * Math.sin((d[1] * 2 * Math.PI) / this.dialogTpNodeCount)
          );
        })
        .on("click", (event, d) => {
          this.pvSwitch(d[0], d[1]);
        });

      //draw connect lines
      this.topologyZone
        .selectAll(".tpUV")
        .data(this.tp)
        .join("line")
        .classed("tpUV", true)
        .attr("stroke", "steelblue")
        .attr("stroke-width", 6)
        .attr("x1", (d) => {
          return (
            150 +
            150 *
              Math.cos(
                (Number(d.split("_")[0]) * 2 * Math.PI) / this.dialogTpNodeCount
              )
          );
        })
        .attr("y1", (d) => {
          return (
            150 +
            150 *
              Math.sin(
                (Number(d.split("_")[0]) * 2 * Math.PI) / this.dialogTpNodeCount
              )
          );
        })
        .attr("x2", (d) => {
          return (
            150 +
            150 *
              Math.cos(
                (Number(d.split("_")[1]) * 2 * Math.PI) / this.dialogTpNodeCount
              )
          );
        })
        .attr("y2", (d) => {
          return (
            150 +
            150 *
              Math.sin(
                (Number(d.split("_")[1]) * 2 * Math.PI) / this.dialogTpNodeCount
              )
          );
        })
        .on("click", (event, d) => {
          this.pvSwitch(Number(d.split("_")[0]), Number(d.split("_")[1]));
        });

      // draw qbit
      let q_root = this.topologyZone
        .selectAll(".qNode")
        .data(this.qbit)
        .join("g")
        .classed("qNode", true)
        .style("transform", (d, i) => {
          return `translateX(${
            150 + 150 * Math.cos((i * 2 * Math.PI) / this.dialogTpNodeCount)
          }px) translateY(${
            150 + 150 * Math.sin((i * 2 * Math.PI) / this.dialogTpNodeCount)
          }px)`;
        });
      q_root
        .append("circle")
        .attr("stroke", "steelblue")
        .attr("stroke-width", 1.5)
        .attr("fill", "lightblue")
        .attr("r", 15);

      q_root
        .append("text")
        .text((d, i) => `${i}`)
        .style("transform", () => {
          return `translateY(5px)`;
        })
        .attr("fill", "black")
        .style("text-anchor", "middle");
    },
    TpNodeCountChange() {
      // 节点数修改
      if (this.dialogTpNodeCount < 0) {
        this.dialogTpNodeCount = 0;
      }
      if (this.dialogTpNodeCount > this.qbit.length) {
        while (this.dialogTpNodeCount > this.qbit.length) {
          this.qbit.push(this.qbit.length);
        }
      } else if (this.dialogTpNodeCount < this.qbit.length) {
        while (this.dialogTpNodeCount < this.qbit.length) {
          let removedQbit = this.qbit.pop();
          for (let k = this.tp.length - 1; k >= 0; k--) {
            let u = Number(this.tp[k].split("_")[0]);
            let v = Number(this.tp[k].split("_")[1]);
            if (u == removedQbit || v == removedQbit) {
              this.tp.splice(k, 1);
            }
          }
        }
      }
      this.fullTp = [];
      for (let i = 0; i < this.qbit.length; i++) {
        for (let j = i + 1; j < this.qbit.length; j++) {
          this.fullTp.push([this.qbit[i], this.qbit[j]]);
        }
      }
      this.updateTopology();
    },
    TpLoad(file) {
      // 加载topology文件
      console.log(file);
      let reader = new FileReader();
      reader.readAsText(file, "UTF-8");

      reader.onload = (evt) => {
        let text = evt.target.result;
        console.log(text);
        let l_text = text.split("\n");
        console.log(l_text);
        let new_tp = [];
        try {
          let new_q = Number(l_text[1]);
          for (let i = 2; i < l_text.length; i++) {
            if (l_text[i].length == 0) {
              continue;
            }
            let u = Number(l_text[i].split(" ")[0]);
            let v = Number(l_text[i].split(" ")[1]);
            new_tp.push(`${u}_${v}`);
          }
          this.dialogTpNodeCount = new_q;
          this.tp = new_tp;
          this.TpNodeCountChange();
        } catch (error) {
          console.log(error);
        }
      };

      reader.onerror = (evt) => {
        console.error(evt);
      };
    },
    TpConfirm() {
      // 通知外层topology变化
      this.$emit("UpdataTopology", this.tp, this.qbit);
    },
    TpCancel() {
      // 取消topology修改
      this.ResetTopology();
      this.ResetQbit();
    },
    ResetTopology() {
      // 重置topology
      this.tp = JSON.parse(JSON.stringify(this.topology));
    },
    ResetQbit() {
      // 重置节点数
      this.qbit = JSON.parse(JSON.stringify(this.q));
      this.fullTp = [];
      this.dialogTpNodeCount = this.qbit.length;
      for (let i = 0; i < this.qbit.length; i++) {
        for (let j = i + 1; j < this.qbit.length; j++) {
          this.fullTp.push([this.qbit[i], this.qbit[j]]);
        }
      }
    },
    saveQCDA() {
      // 通知外层保存当前qasm
      this.$emit("SaveQCDA");
    },
    runQCDA() {
      // 通知外层运行当前qasm
      let setting = {};
      setting.device=this.dialogBe;
      setting.shots=this.dialogSeShots;
      switch(setting.device)
      {
        case 'CPU':
          setting.backend='unitary';
          setting.precision=this.dialogSe_Precision;
          break;
        case 'GPU':
          setting.backend=this.dialogBe_Backend;
          setting.precision=this.dialogSe_Precision;
          switch(this.dialogBe_Backend){
            case 'unitary':
              break;
            case 'statevector':
              setting.gpu_device_id=this.dialogSe_GPU_device_id;
              setting.sync=this.dialogSe_sync;
              setting.optimize=this.dialogSe_optimize;
              break;
            case 'multiGPU':
              setting.ndev=this.dialogSe_ndev;
              setting.sync=this.dialogSe_sync;
              break;
          }
          break;
        case 'qiskit':
          setting.token=this.dialogSeToken;
          break;
        case 'qcompute':
          setting.token=this.dialogSeToken;
          break;
      }
      this.$emit("RunQCDA", this.opSwitch, this.mapSwitch, setting);
    },
    loadQCDA(file) {
      // 通知外层已载入qasm
      this.$emit("LoadQCDA", file);
      return false;
    },
    ChangeSet() {
      // 通知外层切换instruction set
      this.$emit("ChangeSet", this.dialogCmd);
      return false;
    },
    AddToCustomerSet(gate) {
      // 添加gate到customerSet
      this.tempSet.splice(this.tempSet.indexOf(gate), 1);
      this.customerSet.push(gate);
      this.UpdateCustomerSet();
    },
    RemoveFromCustomerSet(gate) {
      // 从customerSet中移除当前gate
      this.customerSet.splice(this.customerSet.indexOf(gate), 1);
      this.tempSet.push(gate);
      this.UpdateCustomerSet();
    },
    UpdateCustomerSet() {
      // 通知外层更新customerSet
      this.$emit("UpdateCustomerSet", this.customerSet);
      if (this.all_sets[this.dialogCmd]["name"] == "CustomerSet") {
        this.ChangeSet();
      }
    },
  },
  mounted: function () {},
  watch: {
    dialogCmd() {
      this.ChangeSet();
    },
    customer_set() {
      this.customerSet = JSON.parse(JSON.stringify(this.customer_set));
      this.fullSet = JSON.parse(JSON.stringify(this.all_sets[0]["gates"]));
      this.tempSet = [];
      this.fullSet.forEach((f_gate) => {
        let in_customer_set = false;
        this.customerSet.forEach((c_gate) => {
          if (c_gate.name == f_gate.name) {
            in_customer_set = true;
          }
        });
        if (!in_customer_set) {
          this.tempSet.push(f_gate);
        }
      });
    },
    topology() {
      this.ResetTopology();
    },
    q() {
      this.ResetQbit();
    },
  },
  emits: {
    SaveQCDA: null,
    RunQCDA: null,
    LoadQCDA: null,
    ChangeSet: null,
    UpdateCustomerSet: null,
    UpdataTopology: null,
  },
};
</script>