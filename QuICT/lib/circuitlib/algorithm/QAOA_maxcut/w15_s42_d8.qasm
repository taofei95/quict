OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
cx q[12], q[11];
rz(-0.8866894841194153) q[11];
cx q[12], q[11];
cx q[1], q[3];
rz(-0.8866894841194153) q[3];
cx q[1], q[3];
cx q[12], q[14];
rz(-0.8866894841194153) q[14];
cx q[12], q[14];
cx q[0], q[2];
rz(-0.8866894841194153) q[2];
cx q[0], q[2];
rx(0.9535027146339417) q[0];
rx(0.9535027146339417) q[1];
rx(0.9535027146339417) q[2];
rx(0.9535027146339417) q[3];
rx(0.9535027146339417) q[4];
rx(0.9535027146339417) q[5];
rx(0.9535027146339417) q[6];
rx(0.9535027146339417) q[7];
rx(0.9535027146339417) q[8];
rx(0.9535027146339417) q[9];
rx(0.9535027146339417) q[10];
rx(0.9535027146339417) q[11];
rx(0.9535027146339417) q[12];
rx(0.9535027146339417) q[13];
rx(0.9535027146339417) q[14];
