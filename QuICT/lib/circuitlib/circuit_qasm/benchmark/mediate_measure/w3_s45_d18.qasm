OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
rz(4.168175207359987) q[0];
rz(5.427246390314631) q[2];
rz(5.915529156208332) q[1];
rz(4.161977869898531) q[2];
cx q[1], q[0];
cx q[0], q[2];
rz(0.09093340589009646) q[1];
rz(5.91512978974229) q[2];
rz(4.583577734220702) q[0];
rz(4.977240810647241) q[1];
rz(5.066182275145685) q[0];
rz(1.4354171575503316) q[2];
rz(0.0972914878645445) q[1];
rz(3.3402794929121544) q[1];
rz(1.224999940853581) q[2];
rz(2.9642302581149234) q[0];
rz(4.288432238211861) q[1];
cx q[0], q[2];
cx q[0], q[1];
rz(1.656718010750099) q[2];
cx q[0], q[1];
rz(4.773331741130928) q[2];
rz(0.38169194688610875) q[1];
rz(5.406020579892045) q[2];
rz(5.238261330767452) q[0];
cx q[0], q[1];
rz(5.209520046890382) q[2];
rz(2.6239251751226775) q[1];
rz(3.209773009189329) q[0];
rz(0.9514381561803894) q[2];
rz(1.443537825777903) q[0];
rz(4.022532503561028) q[1];
rz(4.87997447565829) q[2];
rz(1.1434998437327504) q[1];
rz(1.7011753228880166) q[0];
rz(0.36552034519581883) q[2];
rz(4.384393814719633) q[0];
rz(3.142775168385165) q[2];
rz(5.256036362568107) q[1];
cx q[2], q[0];
rz(3.458660096268258) q[1];
rz(5.925166799148876) q[1];
rz(3.6594971598818016) q[0];
rz(5.720988693504589) q[2];
rz(2.970135799043495) q[1];