OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
cx q[4], q[12];
rz(1.6594201385501732) q[3];
rz(5.22227091794254) q[2];
rz(5.860738327018597) q[0];
rz(2.7860234246371802) q[1];
rz(5.22738756369239) q[5];
rz(3.880315201314243) q[9];
rz(5.768534494370015) q[6];
cx q[10], q[8];
rz(5.09044889832056) q[11];
rz(0.15394859091050703) q[7];
rz(0.4248441103253985) q[2];
rz(4.136191574256396) q[10];
rz(2.385778569937878) q[7];
rz(0.1836001972746821) q[3];
rz(1.891249846383986) q[11];
cx q[4], q[6];
rz(4.781460148857249) q[0];
rz(4.604093152700617) q[9];
rz(3.783932470200951) q[5];
rz(4.729979436400028) q[1];
rz(1.8771726253368701) q[8];
rz(5.88406648283872) q[12];
rz(4.087200591235372) q[4];
rz(3.746954601811734) q[11];
rz(3.921125262883104) q[1];
cx q[6], q[7];
rz(2.0018392609427016) q[0];
rz(3.1394787003199163) q[3];
rz(6.058425293875114) q[2];
cx q[8], q[5];
rz(3.0258711276299937) q[9];
rz(2.8904749832673513) q[12];
rz(3.663257674288245) q[10];
rz(6.137809304515149) q[1];
rz(2.2632628326458604) q[8];
cx q[4], q[9];
rz(0.37830780459453817) q[3];
cx q[12], q[10];
rz(5.68578395497014) q[7];
cx q[5], q[6];
rz(1.250019328880731) q[11];
cx q[0], q[2];
rz(6.122059114265401) q[1];
rz(3.1278306263774427) q[7];
rz(0.2114935400458904) q[6];
rz(5.287158179452975) q[9];
rz(3.9523291856518132) q[3];
rz(4.496226228979487) q[5];
rz(2.2778291302957347) q[11];
cx q[2], q[12];
rz(3.8724970610026204) q[10];
cx q[8], q[0];
rz(5.276479648560211) q[4];
rz(1.0867095698127103) q[1];
rz(5.257048554250201) q[8];
rz(1.8494566282211857) q[4];
cx q[12], q[7];
rz(5.5176824953761185) q[3];
rz(3.4051529846289497) q[6];
rz(1.7250023695643983) q[5];
cx q[0], q[11];
rz(4.695739031197743) q[2];
rz(4.998085190468421) q[10];
rz(1.8970509838569565) q[9];
cx q[12], q[7];
rz(3.796034307114022) q[6];
rz(2.671285283836295) q[8];
rz(4.014987749282748) q[1];
cx q[3], q[4];
rz(2.6647781439107274) q[10];
cx q[5], q[9];
cx q[11], q[2];
rz(0.25990121290242146) q[0];
cx q[2], q[7];
rz(0.18656775498109054) q[12];
rz(1.564119842291912) q[11];
rz(4.072549913428282) q[1];
cx q[0], q[5];
rz(0.027921142556411007) q[8];
rz(5.836282152933076) q[3];
cx q[9], q[4];
rz(2.611338838707983) q[10];
rz(5.901456908042702) q[6];
rz(0.5931936690290217) q[9];
rz(4.388286843617259) q[4];
cx q[11], q[0];
rz(2.995571492303853) q[5];
cx q[12], q[8];
rz(3.291196489550706) q[3];
cx q[1], q[6];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
