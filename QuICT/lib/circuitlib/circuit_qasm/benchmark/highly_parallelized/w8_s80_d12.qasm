OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
rz(2.082201652585647) q[3];
rz(5.5693927006945465) q[0];
rz(3.648774969668128) q[4];
cx q[2], q[7];
rz(6.273455544592127) q[1];
rz(6.037974683757703) q[6];
rz(3.6890721530492794) q[5];
cx q[3], q[6];
rz(3.4757173999041377) q[0];
cx q[1], q[7];
cx q[2], q[4];
rz(2.431333449947655) q[5];
rz(2.4907682477123023) q[2];
cx q[3], q[6];
rz(5.672563568666415) q[0];
rz(3.9943072464517164) q[5];
rz(3.9824767484949786) q[1];
rz(3.679225469082137) q[7];
rz(4.828810067321244) q[4];
rz(0.41631153507767105) q[3];
rz(1.2563809104062076) q[2];
cx q[5], q[0];
rz(1.679756351083143) q[6];
rz(0.9041782961566656) q[1];
rz(0.5127973836860403) q[4];
rz(4.692093013122172) q[7];
rz(1.8162977864619065) q[3];
rz(1.830472604667037) q[5];
rz(2.430334957020476) q[7];
rz(2.9414174802173725) q[4];
rz(1.5500514913045933) q[0];
rz(4.576918453469714) q[6];
rz(2.2056749949238905) q[1];
rz(2.3795347378514995) q[2];
rz(0.3231515199061216) q[4];
rz(5.490110714540753) q[2];
rz(2.878007762121669) q[6];
cx q[5], q[3];
cx q[1], q[7];
rz(3.190708505427569) q[0];
cx q[0], q[7];
rz(0.13404931716992557) q[6];
cx q[4], q[2];
cx q[5], q[3];
rz(3.3435638304177693) q[1];
rz(2.0431388383558207) q[4];
rz(2.8721976461085217) q[5];
cx q[3], q[6];
rz(4.114126015332822) q[7];
cx q[0], q[2];
rz(3.181295355796921) q[1];
rz(3.6697324920105943) q[7];
rz(2.348762966063913) q[2];
cx q[6], q[1];
rz(1.112473763936482) q[5];
rz(2.5453541007261364) q[0];
rz(0.05319149967182679) q[4];
rz(3.290317046411842) q[3];
rz(3.059484095646048) q[1];
rz(3.07513431842567) q[2];
rz(2.0751305745606836) q[3];
rz(3.5062230012513753) q[5];
rz(5.004618364307237) q[7];
rz(1.7870050368173713) q[6];
cx q[4], q[0];
rz(3.485378468634364) q[0];
rz(2.520043573157342) q[6];
rz(1.2169757039675444) q[2];
cx q[7], q[4];
rz(2.335758567335936) q[5];
rz(4.185173775507701) q[1];
rz(0.9557641026118799) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];