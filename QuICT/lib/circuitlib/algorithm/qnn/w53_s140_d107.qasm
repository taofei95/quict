OPENQASM 2.0;
include "qelib1.inc";
qreg q[53];
creg c[53];
x q[0];
x q[4];
x q[6];
x q[8];
x q[9];
x q[11];
x q[12];
x q[13];
x q[15];
x q[17];
x q[18];
x q[19];
x q[20];
x q[21];
x q[23];
x q[24];
x q[25];
x q[27];
x q[28];
x q[29];
x q[30];
x q[32];
x q[34];
x q[35];
x q[36];
x q[37];
x q[38];
x q[39];
x q[42];
x q[44];
x q[45];
x q[48];
x q[51];
x q[0];
h q[0];
rzz(0.6680223345756531) q[0], q[52];
rzz(0.8852099180221558) q[1], q[52];
rzz(0.8609462380409241) q[2], q[52];
rzz(0.9879807233810425) q[3], q[52];
rzz(0.9260596036911011) q[4], q[52];
rzz(0.5871047377586365) q[5], q[52];
rzz(0.37150686979293823) q[6], q[52];
rzz(0.3595152497291565) q[7], q[52];
rzz(0.17695564031600952) q[8], q[52];
rzz(0.07365638017654419) q[9], q[52];
rzz(0.9950640797615051) q[10], q[52];
rzz(0.6707490086555481) q[11], q[52];
rzz(0.6982831358909607) q[12], q[52];
rzz(0.2588222622871399) q[13], q[52];
rzz(0.6584749221801758) q[14], q[52];
rzz(0.10825467109680176) q[15], q[52];
rzz(0.203815758228302) q[16], q[52];
rzz(0.071769118309021) q[17], q[52];
rzz(0.19953805208206177) q[18], q[52];
rzz(0.8257191181182861) q[19], q[52];
rzz(0.18390530347824097) q[20], q[52];
rzz(0.563903272151947) q[21], q[52];
rzz(0.54969722032547) q[22], q[52];
rzz(0.23556530475616455) q[23], q[52];
rzz(0.7105967998504639) q[24], q[52];
rzz(0.8884067535400391) q[25], q[52];
rzz(0.08962851762771606) q[26], q[52];
rzz(0.5497221946716309) q[27], q[52];
rzz(0.10377514362335205) q[28], q[52];
rzz(0.7242958545684814) q[29], q[52];
rzz(0.07227414846420288) q[30], q[52];
rzz(0.026370465755462646) q[31], q[52];
rzz(0.8801000714302063) q[32], q[52];
rzz(0.829860508441925) q[33], q[52];
rzz(0.9753871560096741) q[34], q[52];
rzz(0.6162176132202148) q[35], q[52];
rzz(0.5408759117126465) q[36], q[52];
rzz(0.12614715099334717) q[37], q[52];
rzz(0.42590487003326416) q[38], q[52];
rzz(0.523354709148407) q[39], q[52];
rzz(0.4107300639152527) q[40], q[52];
rzz(0.9353851079940796) q[41], q[52];
rzz(0.779202938079834) q[42], q[52];
rzz(0.17500418424606323) q[43], q[52];
rzz(0.05499941110610962) q[44], q[52];
rzz(0.3333315849304199) q[45], q[52];
rzz(0.5360687375068665) q[46], q[52];
rzz(0.4810338020324707) q[47], q[52];
rzz(0.9704304337501526) q[48], q[52];
rzz(0.7258064150810242) q[49], q[52];
rzz(0.5170546770095825) q[50], q[52];
rzz(0.3738224506378174) q[51], q[52];
rzz(0.9112953543663025) q[0], q[52];
rzz(0.2197643518447876) q[1], q[52];
rzz(0.6811544299125671) q[2], q[52];
rzz(0.7084458470344543) q[3], q[52];
rzz(0.6697544455528259) q[4], q[52];
rzz(0.7642578482627869) q[5], q[52];
rzz(0.561923623085022) q[6], q[52];
rzz(0.4034026265144348) q[7], q[52];
rzz(0.5293595194816589) q[8], q[52];
rzz(0.9126337170600891) q[9], q[52];
rzz(0.609634280204773) q[10], q[52];
rzz(0.5279523730278015) q[11], q[52];
rzz(0.6051191091537476) q[12], q[52];
rzz(0.8339897394180298) q[13], q[52];
rzz(0.825038731098175) q[14], q[52];
rzz(0.11775845289230347) q[15], q[52];
rzz(0.6876066327095032) q[16], q[52];
rzz(0.5190452337265015) q[17], q[52];
rzz(0.7536632418632507) q[18], q[52];
rzz(0.045656800270080566) q[19], q[52];
rzz(0.03817284107208252) q[20], q[52];
rzz(0.22161269187927246) q[21], q[52];
rzz(0.47102344036102295) q[22], q[52];
rzz(0.9895716309547424) q[23], q[52];
rzz(0.6181480884552002) q[24], q[52];
rzz(0.7011862397193909) q[25], q[52];
rzz(0.48688918352127075) q[26], q[52];
rzz(0.7964714169502258) q[27], q[52];
rzz(0.6760263442993164) q[28], q[52];
rzz(0.37478750944137573) q[29], q[52];
rzz(0.9485838413238525) q[30], q[52];
rzz(0.239152729511261) q[31], q[52];
rzz(0.0068239569664001465) q[32], q[52];
rzz(0.14989137649536133) q[33], q[52];
rzz(0.1378372311592102) q[34], q[52];
rzz(0.9675108790397644) q[35], q[52];
rzz(0.20373600721359253) q[36], q[52];
rzz(0.825808584690094) q[37], q[52];
rzz(0.7398805022239685) q[38], q[52];
rzz(0.06240487098693848) q[39], q[52];
rzz(0.6164005398750305) q[40], q[52];
rzz(0.5524119734764099) q[41], q[52];
rzz(0.7540640234947205) q[42], q[52];
rzz(0.6090071201324463) q[43], q[52];
rzz(0.878968358039856) q[44], q[52];
rzz(0.6992396116256714) q[45], q[52];
rzz(0.4838378429412842) q[46], q[52];
rzz(0.02232182025909424) q[47], q[52];
rzz(0.033154428005218506) q[48], q[52];
rzz(0.04720944166183472) q[49], q[52];
rzz(0.23932820558547974) q[50], q[52];
rzz(0.6587642431259155) q[51], q[52];
h q[0];