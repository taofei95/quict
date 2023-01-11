OPENQASM 2.0;
include "qelib1.inc";
qreg q[50];
creg c[50];
x q[1];
x q[5];
x q[8];
x q[9];
x q[12];
x q[13];
x q[20];
x q[23];
x q[24];
x q[25];
x q[26];
x q[27];
x q[28];
x q[29];
x q[30];
x q[31];
x q[33];
x q[34];
x q[37];
x q[39];
x q[41];
x q[44];
x q[47];
x q[48];
x q[0];
h q[0];
ryy(0.8437568545341492) q[0], q[49];
ryy(0.4135262966156006) q[1], q[49];
ryy(0.6028174757957458) q[2], q[49];
ryy(0.17857688665390015) q[3], q[49];
ryy(0.9227702617645264) q[4], q[49];
ryy(0.4265453815460205) q[5], q[49];
ryy(0.14190667867660522) q[6], q[49];
ryy(0.8136272430419922) q[7], q[49];
ryy(0.14096051454544067) q[8], q[49];
ryy(0.8673809766769409) q[9], q[49];
ryy(0.9061349630355835) q[10], q[49];
ryy(0.954811692237854) q[11], q[49];
ryy(0.7844679355621338) q[12], q[49];
ryy(0.004663944244384766) q[13], q[49];
ryy(0.6323443651199341) q[14], q[49];
ryy(0.8128437399864197) q[15], q[49];
ryy(0.8909300565719604) q[16], q[49];
ryy(0.5995246171951294) q[17], q[49];
ryy(0.01624774932861328) q[18], q[49];
ryy(0.937663197517395) q[19], q[49];
ryy(0.36800479888916016) q[20], q[49];
ryy(0.9486134648323059) q[21], q[49];
ryy(0.8010897040367126) q[22], q[49];
ryy(0.36169469356536865) q[23], q[49];
ryy(0.22007668018341064) q[24], q[49];
ryy(0.770452618598938) q[25], q[49];
ryy(0.8116058111190796) q[26], q[49];
ryy(0.39440852403640747) q[27], q[49];
ryy(0.9803217649459839) q[28], q[49];
ryy(0.12220001220703125) q[29], q[49];
ryy(0.5212355852127075) q[30], q[49];
ryy(0.3310902714729309) q[31], q[49];
ryy(0.8547820448875427) q[32], q[49];
ryy(0.963091254234314) q[33], q[49];
ryy(0.5823631882667542) q[34], q[49];
ryy(0.9443464875221252) q[35], q[49];
ryy(0.9184112548828125) q[36], q[49];
ryy(0.1447913646697998) q[37], q[49];
ryy(0.8995833992958069) q[38], q[49];
ryy(0.9178946614265442) q[39], q[49];
ryy(0.09946483373641968) q[40], q[49];
ryy(0.07194602489471436) q[41], q[49];
ryy(0.5109673738479614) q[42], q[49];
ryy(0.047553181648254395) q[43], q[49];
ryy(0.5922011733055115) q[44], q[49];
ryy(0.9903733134269714) q[45], q[49];
ryy(0.06257688999176025) q[46], q[49];
ryy(0.49004584550857544) q[47], q[49];
ryy(0.9565462470054626) q[48], q[49];
rzx(0.6609853506088257) q[0], q[49];
rzx(0.9089956879615784) q[1], q[49];
rzx(0.89010089635849) q[2], q[49];
rzx(0.9475123882293701) q[3], q[49];
rzx(0.6913091540336609) q[4], q[49];
rzx(0.7690754532814026) q[5], q[49];
rzx(0.06444966793060303) q[6], q[49];
rzx(0.2217710018157959) q[7], q[49];
rzx(0.14372706413269043) q[8], q[49];
rzx(0.5797688364982605) q[9], q[49];
rzx(0.5380721092224121) q[10], q[49];
rzx(0.8955209255218506) q[11], q[49];
rzx(0.8627658486366272) q[12], q[49];
rzx(0.39155757427215576) q[13], q[49];
rzx(0.44115203619003296) q[14], q[49];
rzx(0.6341134905815125) q[15], q[49];
rzx(0.663898229598999) q[16], q[49];
rzx(0.11149293184280396) q[17], q[49];
rzx(0.39047861099243164) q[18], q[49];
rzx(0.8822869658470154) q[19], q[49];
rzx(0.741144597530365) q[20], q[49];
rzx(0.9487879276275635) q[21], q[49];
rzx(0.10615426301956177) q[22], q[49];
rzx(0.7552198171615601) q[23], q[49];
rzx(0.8260319232940674) q[24], q[49];
rzx(0.3041677474975586) q[25], q[49];
rzx(0.10190439224243164) q[26], q[49];
rzx(0.4463338255882263) q[27], q[49];
rzx(0.06323188543319702) q[28], q[49];
rzx(0.31326329708099365) q[29], q[49];
rzx(0.6653119921684265) q[30], q[49];
rzx(0.5772725939750671) q[31], q[49];
rzx(0.07716411352157593) q[32], q[49];
rzx(0.5652183294296265) q[33], q[49];
rzx(0.26652050018310547) q[34], q[49];
rzx(0.28013867139816284) q[35], q[49];
rzx(0.9550812244415283) q[36], q[49];
rzx(0.7549905180931091) q[37], q[49];
rzx(0.3896419405937195) q[38], q[49];
rzx(0.5405530333518982) q[39], q[49];
rzx(0.6383386254310608) q[40], q[49];
rzx(0.727345883846283) q[41], q[49];
rzx(0.3607556223869324) q[42], q[49];
rzx(0.12602436542510986) q[43], q[49];
rzx(0.5987079739570618) q[44], q[49];
rzx(0.37006843090057373) q[45], q[49];
rzx(0.5587384700775146) q[46], q[49];
rzx(0.7936181426048279) q[47], q[49];
rzx(0.29564201831817627) q[48], q[49];
h q[0];
