OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
x q[2];
x q[3];
x q[4];
x q[5];
x q[0];
h q[0];
rzz(0.9935101866722107) q[0], q[6];
rzz(0.20438647270202637) q[1], q[6];
rzz(0.9928470253944397) q[2], q[6];
rzz(0.09547537565231323) q[3], q[6];
rzz(0.029439330101013184) q[4], q[6];
rzz(0.16709774732589722) q[5], q[6];
rzz(0.6506385207176208) q[0], q[6];
rzz(0.953069806098938) q[1], q[6];
rzz(0.5707724690437317) q[2], q[6];
rzz(0.0045882463455200195) q[3], q[6];
rzz(0.4952126145362854) q[4], q[6];
rzz(0.6146045327186584) q[5], q[6];
h q[0];
