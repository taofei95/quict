OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rz(5.98019783904531) q[1];
rz(5.968969581532458) q[2];
rz(0.7044962077857564) q[4];
rz(1.8456865068511263) q[0];
rz(0.6755414518029652) q[3];
rz(3.603118108157645) q[4];
cx q[3], q[2];
rz(1.5583589908701743) q[1];
rz(0.11421042289383317) q[0];
rz(0.9635524647676618) q[1];
rz(4.477524702287822) q[2];
rz(5.550587133300238) q[0];
rz(4.388768045575762) q[3];
rz(0.4540674035992287) q[4];
rz(0.8815129784021724) q[4];
rz(2.449611790974183) q[1];
rz(1.8465834620455157) q[3];
rz(0.5785212634987367) q[0];
rz(1.7374518067386981) q[2];
cx q[2], q[1];
rz(4.77265246810831) q[3];
rz(3.936676462296503) q[4];
rz(3.1843124910751754) q[0];
rz(5.016360737212403) q[1];
rz(4.713540667474667) q[4];
cx q[0], q[2];
rz(1.048157584553306) q[3];
rz(4.175593574893456) q[2];
rz(5.10802833070183) q[0];
rz(6.023259369814625) q[3];
rz(4.532233577900441) q[4];
rz(0.017055484832440883) q[1];
rz(0.8691473075252489) q[3];
cx q[2], q[4];
rz(2.444477735113258) q[1];
rz(0.3321937100409738) q[0];
rz(0.5622114206369871) q[3];
rz(1.5312011124692526) q[2];
rz(6.165415455245529) q[0];
rz(0.5768458484056034) q[1];
rz(0.671126450177133) q[4];
rz(1.3854937104846108) q[0];
cx q[3], q[1];
rz(6.215813903618736) q[4];
rz(4.267344525289156) q[2];
rz(3.305206276387974) q[1];
rz(0.7479810396690806) q[3];
rz(4.352760667334549) q[0];
rz(3.7085888268602227) q[4];
rz(5.33209473461132) q[2];
rz(5.866022806930209) q[2];
rz(0.9955433997646507) q[1];
rz(2.200097391547563) q[3];
rz(2.566098087418326) q[0];
rz(6.123431411968191) q[4];
rz(6.008028755905215) q[3];
rz(1.3211205316858754) q[1];
rz(2.283011674547184) q[2];
rz(4.352385797474213) q[4];
rz(1.066684130211626) q[0];
rz(3.8182086958806893) q[1];
cx q[2], q[4];
rz(2.298988974331358) q[3];
rz(4.263078087818187) q[0];
rz(1.7328464537586665) q[3];
rz(6.06374275506642) q[1];
rz(6.080068937576874) q[0];
rz(4.842881458681309) q[2];
rz(4.0757347316845385) q[4];
rz(2.6748917846537523) q[0];
rz(2.1353255279507226) q[2];
rz(2.5697441479982293) q[3];
cx q[4], q[1];
cx q[2], q[3];
rz(4.1341932121459735) q[1];
rz(4.255248874017663) q[4];
rz(4.663886601170415) q[0];
rz(2.968728154981735) q[4];
rz(5.732324209235831) q[3];
rz(5.801643896962914) q[1];
rz(2.341076900894348) q[0];
rz(5.53190672397545) q[2];
rz(5.367014059132087) q[3];
rz(3.235291074576355) q[1];
cx q[0], q[4];
rz(5.749387732342972) q[2];
rz(5.142848051171018) q[4];
rz(0.4002901127119434) q[0];
cx q[1], q[2];
rz(1.3896882522446208) q[3];
rz(2.7658777829513683) q[0];
cx q[3], q[2];
rz(5.659910211252747) q[4];
rz(5.19468350831328) q[1];
rz(0.6883657211973898) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
