OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rz(2.5299408338346456) q[5];
rz(3.609115460611458) q[2];
rz(4.581658187881203) q[0];
rz(1.4009932455290606) q[6];
rz(3.8951587570424593) q[3];
rz(2.5111955684472207) q[8];
rz(1.647647190094902) q[9];
cx q[1], q[4];
rz(4.216749133945264) q[7];
rz(4.647621784892783) q[1];
rz(6.25434729854629) q[4];
rz(2.5324975962027776) q[5];
rz(4.315764999356811) q[2];
rz(3.1386253431344167) q[6];
cx q[8], q[7];
rz(3.5327685668566353) q[9];
rz(0.5515861724247216) q[3];
rz(2.9890643013448073) q[0];
rz(1.4380347335381913) q[7];
cx q[5], q[1];
rz(5.4191970630292685) q[4];
cx q[2], q[3];
rz(3.716922626666572) q[9];
rz(5.647982508693588) q[0];
rz(2.2548291995879914) q[6];
rz(1.768309410002178) q[8];
rz(1.0255121295224607) q[4];
rz(3.0798912336719475) q[1];
rz(2.6846544805878674) q[0];
rz(5.809497167950145) q[6];
rz(1.4443878477738514) q[8];
rz(2.5120580278133047) q[7];
rz(6.280602793638893) q[9];
rz(5.81028125884456) q[2];
rz(5.520752565043224) q[5];
rz(5.76951994864795) q[3];
rz(3.9690504572283727) q[9];
rz(3.668250848013822) q[1];
rz(3.9524925788582115) q[6];
rz(3.924280474826027) q[0];
rz(6.1627199408094295) q[8];
rz(0.4238907535920652) q[5];
rz(4.023256455027156) q[7];
rz(4.429581504026692) q[3];
cx q[2], q[4];
cx q[8], q[6];
rz(4.848788085311185) q[0];
cx q[7], q[9];
rz(1.6192800832497833) q[2];
rz(3.3812174538594872) q[3];
rz(4.639934426152851) q[5];
rz(2.7572926702133636) q[1];
rz(1.1648220513013807) q[4];
cx q[5], q[8];
rz(3.031005226227589) q[0];
rz(0.040243053741226845) q[7];
cx q[3], q[6];
rz(1.9238565115663953) q[2];
rz(1.7909234390108337) q[4];
rz(3.868515987411247) q[1];
rz(2.651496177929876) q[9];
rz(2.003810218304433) q[7];
rz(0.11614495773318156) q[1];
rz(5.010632590311854) q[3];
cx q[8], q[2];
rz(3.650272545576435) q[4];
rz(1.0254734863157937) q[6];
rz(5.287490254148688) q[5];
rz(4.8090698161729595) q[0];
rz(5.9866532099639205) q[9];
rz(4.458992143243684) q[3];
rz(3.8782165009474263) q[6];
rz(1.88456510828431) q[2];
rz(3.1158498035986484) q[5];
rz(1.7594438048063568) q[8];
rz(1.2034470348422222) q[4];
rz(5.744464930010179) q[0];
rz(0.7930936154453065) q[7];
rz(5.028304644971231) q[1];
rz(6.2325937721812155) q[9];
rz(5.620184108570153) q[4];
rz(2.313257740397371) q[0];
rz(1.3356936947572) q[7];
cx q[3], q[6];
rz(5.081353235180897) q[2];
rz(5.60212417075346) q[9];
rz(2.3233803202896035) q[8];
rz(3.025966951173811) q[5];
rz(3.3702373537769) q[1];
rz(1.4329943492310964) q[4];
rz(5.06938381386805) q[7];
rz(3.8381693793591207) q[0];
rz(5.540022751212311) q[6];
rz(1.0474780743162806) q[8];
rz(2.792151808310311) q[1];
cx q[2], q[5];
rz(5.718498677998774) q[3];
rz(4.243193396878891) q[9];
rz(3.1389017978093676) q[5];
rz(1.9487986843391842) q[9];
cx q[6], q[1];
rz(5.305733863435727) q[2];
rz(0.15528576498092517) q[7];
rz(1.9177209632199452) q[3];
rz(2.905508770684263) q[4];
cx q[8], q[0];
cx q[2], q[3];
rz(1.1094231703768178) q[1];
rz(6.068587103175275) q[9];
rz(3.460596247818734) q[0];
cx q[8], q[5];
rz(2.549413804110196) q[4];
rz(5.177309148816259) q[6];
rz(4.0179440361262015) q[7];
rz(2.6704595682642664) q[2];
rz(4.625086018961752) q[0];
rz(5.643772085245342) q[8];
rz(0.015123975462143446) q[6];
rz(5.152366676247242) q[9];
cx q[3], q[5];
cx q[7], q[4];
rz(1.4891256650803968) q[1];
rz(2.7136545040052655) q[1];
cx q[2], q[7];
cx q[9], q[4];
rz(0.1865471948999743) q[3];
rz(1.6602123257128731) q[8];
cx q[6], q[0];
rz(2.0399272334731497) q[5];
rz(4.081385866106293) q[2];
rz(1.3428502975702592) q[0];
rz(3.1664563802423404) q[8];
rz(4.943609657129301) q[9];
rz(1.8898835681588062) q[1];
rz(5.142012146640293) q[4];
cx q[5], q[3];
rz(1.312939878494873) q[6];
rz(0.052167072726353764) q[7];
rz(6.110209510111205) q[6];
rz(3.7436517966340217) q[8];
rz(4.954279871500702) q[1];
cx q[2], q[4];
rz(0.5543752815220011) q[3];
cx q[9], q[5];
cx q[7], q[0];
rz(1.1633473475308058) q[6];
rz(2.591342291492378) q[2];
rz(2.008697244100768) q[8];
rz(1.6498535423590832) q[7];
rz(4.8948531417810495) q[4];
rz(1.794256452632865) q[9];
rz(2.0389344016270026) q[5];
rz(5.1892437000045994) q[3];
rz(0.6338384048480229) q[1];
rz(0.9831350193992977) q[0];
rz(6.256273116943836) q[6];
rz(1.0348955391932984) q[4];
rz(3.0182645840491413) q[8];
rz(0.7457848278103215) q[7];
cx q[9], q[5];
cx q[0], q[1];
cx q[3], q[2];
rz(1.4587677343965817) q[2];
rz(3.445280444344957) q[7];
rz(0.07094383050618133) q[5];
rz(2.9547988692278797) q[3];
rz(5.881535644508207) q[4];
rz(4.450847658127645) q[8];
rz(2.6778963679174304) q[1];
rz(2.4418492783265937) q[6];
rz(3.633124325344723) q[9];
rz(0.6778725131420353) q[0];
rz(6.098067924194764) q[0];
rz(2.148950377963295) q[4];
rz(4.295703920678747) q[5];
rz(1.6943748270411436) q[2];
rz(0.41606059037808363) q[3];
rz(4.006805183175184) q[7];
rz(0.8182751882627817) q[9];
rz(0.6746670477112051) q[8];
rz(2.5856823736768657) q[1];
rz(4.403638328624563) q[6];
rz(4.291944474156156) q[0];
rz(4.084602146676937) q[4];
rz(4.3563504045234085) q[5];
rz(2.407795283707789) q[3];
rz(4.448291903693761) q[9];
rz(0.19477760549697573) q[7];
cx q[2], q[6];
rz(2.8292839730645203) q[8];
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