OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[27];
rz(6.056614202738501) q[26];
rz(4.184392562959133) q[19];
rz(4.329915749415559) q[18];
rz(1.4287825177015754) q[10];
rz(1.1328571058362529) q[25];
rz(0.09057950988980529) q[9];
rz(0.9136334649553767) q[2];
rz(5.438702471320504) q[11];
rz(2.9156388787787533) q[8];
rz(2.101334691245381) q[3];
rz(1.7466724525196584) q[15];
rz(4.821379521787326) q[14];
rz(0.05885425799198343) q[13];
rz(4.605465771773718) q[20];
cx q[22], q[12];
rz(2.896533783089508) q[24];
rz(3.197328758815687) q[7];
rz(0.059438584374725136) q[17];
rz(4.505846708124934) q[21];
rz(2.9520824527792247) q[16];
rz(0.8519081733798975) q[5];
rz(3.2383919911110848) q[4];
rz(0.42837373106098875) q[1];
rz(2.9640617362397395) q[0];
rz(5.047435856829036) q[23];
rz(5.6549698366809285) q[6];
rz(2.3765395113427923) q[20];
rz(4.513734278064841) q[13];
rz(5.047119852407077) q[12];
rz(0.6181292784871346) q[9];
rz(1.8027804531404257) q[14];
rz(4.34308892093811) q[24];
rz(2.7677970971324113) q[3];
rz(5.152869393420186) q[1];
rz(0.35686990912204564) q[0];
rz(4.237745785222683) q[19];
rz(0.5294694036807019) q[15];
rz(5.742128817557715) q[10];
rz(3.6736345780266726) q[5];
cx q[25], q[17];
rz(5.110847468015498) q[23];
cx q[2], q[4];
rz(3.4944787902493806) q[16];
rz(3.404902860194498) q[7];
rz(1.600981841856088) q[21];
rz(2.821209978405319) q[18];
rz(5.0564995378815265) q[6];
rz(4.988389863977525) q[22];
rz(5.2063668286999185) q[26];
rz(2.5862776384036357) q[11];
rz(1.2233713070866727) q[8];
rz(2.8043776496069293) q[25];
rz(0.09413593303193495) q[11];
rz(3.1774057871479973) q[26];
rz(4.7240203951071855) q[7];
rz(3.190332800335399) q[17];
rz(1.2111021780624294) q[16];
rz(1.3959449421653103) q[2];
cx q[5], q[20];
rz(0.2738396192774727) q[3];
rz(0.0873543440003983) q[1];
cx q[23], q[4];
rz(1.9752609824149427) q[18];
cx q[22], q[10];
cx q[15], q[13];
rz(4.365050817474235) q[9];
rz(2.0685925622810366) q[6];
rz(3.0863844053457616) q[8];
cx q[24], q[21];
cx q[19], q[12];
rz(5.677144008430298) q[14];
rz(6.2142870881085015) q[0];
rz(3.506811011488037) q[11];
rz(5.614571382982679) q[6];
rz(6.024175449507885) q[15];
rz(4.960675780092445) q[18];
rz(5.905402790343316) q[7];
rz(3.3428462519405775) q[19];
rz(3.5581391347432527) q[22];
rz(2.0808816938115817) q[10];
rz(2.508047495358808) q[4];
rz(2.232570834675658) q[20];
rz(1.8560712662790535) q[26];
rz(3.477189727783772) q[17];
rz(5.668499053450553) q[3];
rz(2.3572659203350192) q[23];
rz(3.2262225199162944) q[25];
rz(4.597595829158674) q[1];
cx q[21], q[12];
rz(2.89509152642646) q[9];
rz(4.939735630725702) q[13];
rz(0.4361943847413224) q[24];
rz(5.317970621994945) q[16];
rz(3.261228318713682) q[0];
rz(1.5999676058464225) q[5];
rz(5.147742190195437) q[14];
rz(3.8724474749888933) q[2];
rz(5.871576824262265) q[8];
rz(4.58495788278456) q[14];
rz(0.6801454143825838) q[21];
rz(2.700379988165319) q[17];
cx q[6], q[24];
rz(3.9054684666777777) q[8];
rz(2.1055513325532917) q[4];
cx q[18], q[2];
rz(2.937067445909221) q[12];
rz(2.32695277149705) q[19];
rz(1.702540974134721) q[26];
rz(4.733631348301405) q[5];
rz(1.5471215315326972) q[25];
rz(1.756851342970149) q[10];
rz(4.836847210602159) q[7];
rz(5.309465609408101) q[0];
rz(1.32965316589351) q[9];
rz(4.028323700939109) q[1];
rz(5.662315140937686) q[3];
cx q[20], q[13];
rz(5.856707876908541) q[22];
rz(2.626511506902957) q[16];
rz(4.465956067453535) q[23];
rz(1.472680310806175) q[15];
rz(4.6056967698785565) q[11];
rz(1.669310128362463) q[16];
rz(4.331682853903075) q[1];
cx q[9], q[19];
rz(2.832992946140882) q[21];
cx q[13], q[25];
rz(5.9179943716192) q[24];
rz(0.8217512539135176) q[6];
cx q[15], q[17];
rz(2.3648720104672725) q[14];
rz(3.156068094005742) q[10];
rz(3.299681017840195) q[5];
rz(0.21574914748515397) q[23];
rz(5.8857818877248365) q[0];
rz(3.4922150353451915) q[20];
rz(3.883319262267305) q[11];
rz(3.7112974005217474) q[7];
cx q[2], q[26];
cx q[8], q[22];
cx q[12], q[18];
rz(3.135303954501497) q[4];
rz(0.8611795268842226) q[3];
rz(1.017609542322916) q[23];
rz(1.756208432931598) q[2];
cx q[19], q[26];
rz(0.11130234803798345) q[3];
cx q[15], q[9];
rz(1.1848056282758084) q[25];
rz(0.23961151260485364) q[13];
rz(4.897997433895271) q[16];
cx q[21], q[12];
cx q[8], q[7];
rz(4.549644859989447) q[10];
rz(6.220525970565925) q[14];
rz(4.576337331113124) q[0];
rz(3.3230218963121017) q[5];
rz(1.8837519685070332) q[20];
rz(0.3129987386189874) q[24];
rz(5.604333777636082) q[22];
rz(0.7735939901773193) q[17];
rz(1.150070644087322) q[1];
rz(5.518557868219233) q[11];
rz(3.568210855622928) q[18];
rz(3.118603753165854) q[6];
rz(4.812502044372301) q[4];
rz(1.8647429916576495) q[8];
rz(0.5941175331282023) q[17];
rz(1.4462701342055186) q[26];
rz(3.8729002896588547) q[21];
rz(1.7116219028166149) q[25];
rz(1.0520888236757309) q[11];
rz(5.8209489465932505) q[4];
rz(5.307443223957557) q[10];
rz(1.6799035227257066) q[5];
rz(5.483971152978995) q[14];
rz(2.1886014737465667) q[16];
rz(5.565658370703522) q[9];
rz(0.9154202838559674) q[12];
cx q[13], q[19];
rz(0.2355314905186247) q[15];
rz(5.115818426500885) q[18];
rz(0.441227074435913) q[20];
rz(0.25189539741503125) q[6];
cx q[7], q[3];
rz(2.544353242241996) q[24];
rz(0.6892312113726897) q[23];
rz(4.758792291419236) q[1];
rz(5.408138953747556) q[2];
cx q[0], q[22];
rz(3.138735524888705) q[12];
cx q[0], q[22];
cx q[19], q[21];
rz(4.386882441136244) q[7];
cx q[2], q[9];
rz(5.374517638322679) q[11];
rz(2.6719814988934996) q[26];
rz(2.6894088215154253) q[4];
rz(1.6529190673467522) q[17];
rz(1.6443374668569937) q[23];
rz(2.053953181485558) q[15];
rz(1.5736229865334475) q[5];
rz(6.133857754437488) q[8];
rz(4.455175855947582) q[20];
cx q[14], q[10];
rz(4.29482087848753) q[1];
rz(4.502253200941712) q[13];
rz(5.502402526290497) q[25];
rz(3.043420585923547) q[24];
rz(0.6640190136973565) q[16];
cx q[6], q[18];
rz(0.7770150165096807) q[3];
rz(3.3791256250258273) q[12];
rz(6.031651877911652) q[11];
rz(3.6542245265758324) q[22];
rz(4.190278016578702) q[14];
rz(5.825677745260164) q[1];
rz(4.853453308134578) q[21];
cx q[0], q[17];
rz(0.039346549723904986) q[25];
cx q[3], q[8];
rz(5.1785978364734575) q[6];
cx q[4], q[13];
rz(2.172512087689963) q[5];
rz(4.018612165722093) q[20];
rz(0.5613496539779864) q[7];
rz(2.29903856945132) q[2];
rz(2.5723409307656504) q[19];
rz(3.2731410835999) q[26];
cx q[24], q[10];
rz(2.51966888107379) q[16];
rz(0.5158757051180273) q[15];
rz(3.949734017118416) q[9];
rz(2.770515924350888) q[18];
rz(1.0246233115385122) q[23];
rz(2.5749571735496675) q[8];
rz(4.300408880374592) q[13];
cx q[9], q[6];
cx q[20], q[10];
cx q[21], q[24];
rz(0.5535096718417581) q[4];
cx q[5], q[16];
rz(4.233738166060977) q[17];
cx q[1], q[2];
rz(4.532547695975959) q[7];
rz(4.676893671317724) q[22];
rz(4.7301621463144965) q[3];
rz(4.560678541269778) q[0];
rz(1.00640747905649) q[12];
rz(4.238168453069873) q[18];
rz(2.5515390783609266) q[14];
cx q[11], q[19];
rz(0.6413230824171284) q[25];
rz(1.1086768753332097) q[23];
rz(1.2882979336860894) q[26];
rz(6.071180738864323) q[15];
rz(3.877556677928999) q[20];
rz(0.9624921206267797) q[12];
rz(1.6513021071616185) q[14];
rz(4.546162664843909) q[23];
rz(5.90527482883676) q[6];
rz(2.459860851104217) q[17];
rz(5.641241188239388) q[10];
rz(0.8742994077359593) q[15];
rz(1.042128688778048) q[4];
rz(4.473509528108014) q[7];
cx q[2], q[3];
rz(5.389620098519896) q[16];
rz(1.2996941503211892) q[5];
rz(5.172859482210262) q[11];
rz(0.4362446706797701) q[25];
rz(1.7872023211721662) q[22];
rz(4.4654905802516796) q[1];
rz(5.960140475409212) q[26];
rz(4.590791286247598) q[19];
rz(3.669872207125038) q[18];
rz(2.9467720514826627) q[0];
cx q[21], q[13];
rz(0.05906537510618998) q[9];
rz(3.6042320187720804) q[8];
rz(1.4234374696274665) q[24];
cx q[26], q[21];
rz(2.0153807770652064) q[9];
cx q[22], q[6];
rz(0.6205112625601437) q[1];
rz(3.629023196785287) q[3];
rz(0.5267103569657503) q[13];
rz(1.6865022659431017) q[24];
rz(3.4649449512639885) q[0];
rz(4.4034862394813485) q[23];
cx q[8], q[25];
rz(6.26893989810072) q[7];
rz(3.5211193019700007) q[17];
rz(0.5571428243447923) q[11];
rz(2.0323660208218506) q[14];
rz(0.5788796749497986) q[2];
rz(0.10044344785572573) q[12];
rz(0.021950536023611477) q[15];
rz(3.920651473651289) q[5];
cx q[19], q[18];
rz(5.674614960122608) q[20];
rz(1.2413984269606002) q[10];
rz(5.96612624127057) q[4];
rz(4.489314337032989) q[16];
rz(5.776583919603407) q[21];
rz(0.2660448581184357) q[9];
rz(5.690166830949112) q[24];
rz(2.3875543776389736) q[10];
rz(4.0987517783285865) q[7];
rz(0.6911597375502424) q[19];
rz(5.93657939176902) q[0];
rz(5.893427053796524) q[1];
rz(6.14964401066805) q[13];
cx q[15], q[18];
cx q[20], q[17];
rz(6.094287989694328) q[11];
cx q[2], q[23];
rz(4.844079058621165) q[26];
cx q[8], q[25];
rz(4.325903160083655) q[4];
rz(5.3329647327456895) q[22];
rz(3.905209754654378) q[5];
rz(0.24827619765072861) q[14];
cx q[12], q[6];
cx q[3], q[16];
rz(4.877066828157901) q[21];
cx q[17], q[5];
rz(3.853076986238773) q[6];
rz(4.843936082893745) q[14];
cx q[10], q[15];
rz(2.2057192506732797) q[13];
cx q[2], q[22];
rz(3.6831314032769225) q[16];
rz(3.1179065747561925) q[3];
rz(0.44090959031572546) q[26];
rz(4.45447521273538) q[20];
rz(6.164861120286723) q[18];
rz(6.223112623023006) q[19];
cx q[24], q[25];
rz(6.064699996640107) q[8];
rz(4.697169568905628) q[1];
rz(1.223781987323845) q[7];
rz(2.8419464052479344) q[12];
cx q[11], q[23];
rz(3.2105507079682805) q[0];
cx q[4], q[9];
rz(3.181608554630961) q[10];
cx q[16], q[22];
rz(2.9523269872094344) q[12];
rz(3.239128878558024) q[6];
rz(2.6534087371695123) q[21];
rz(3.9209906216619808) q[18];
rz(4.233737091127039) q[23];
rz(0.2024480658622257) q[15];
rz(5.860817562084404) q[0];
rz(5.416114912173766) q[17];
rz(3.356253051697384) q[19];
rz(6.040121908515957) q[7];
cx q[24], q[1];
rz(3.9519812975114927) q[2];
rz(4.450712793474828) q[5];
cx q[4], q[26];
rz(3.755542432841286) q[8];
rz(4.73930351054027) q[20];
cx q[13], q[11];
rz(4.5094723332364355) q[25];
rz(2.1824422199749005) q[9];
rz(1.962886707351676) q[3];
rz(2.3621051613758417) q[14];
cx q[21], q[24];
rz(3.019186342300175) q[15];
rz(1.8454779216540067) q[11];
rz(3.245841856995681) q[12];
rz(2.635900443241683) q[2];
rz(5.36623879669971) q[7];
rz(2.349638094295347) q[1];
rz(0.039216877301406446) q[4];
rz(4.509894899002315) q[23];
rz(1.920032384699471) q[26];
cx q[10], q[16];
rz(5.168065907236356) q[22];
rz(2.459765957461215) q[5];
rz(0.5964964763709903) q[17];
rz(0.936062435606112) q[13];
rz(0.46713437690248916) q[19];
rz(4.405995334788904) q[8];
rz(4.162079486414078) q[9];
rz(2.4582581115276914) q[0];
rz(3.7776120344259283) q[6];
cx q[18], q[25];
rz(2.0490676057616883) q[20];
rz(2.2885996688444803) q[3];
rz(2.33333972706159) q[14];
rz(5.533382485483291) q[6];
rz(3.4325193941586236) q[12];
rz(0.07805621118205833) q[23];
rz(4.685435162236707) q[5];
rz(5.383079951164653) q[18];
rz(5.424909019320979) q[16];
rz(0.7442512715661139) q[17];
rz(1.492893222850884) q[8];
rz(2.0418035263882373) q[25];
rz(0.8646015663658483) q[21];
rz(3.195569601393066) q[2];
rz(3.0150769527144226) q[13];
rz(3.539210927046843) q[1];
cx q[11], q[10];
cx q[9], q[20];
rz(3.1615037550829888) q[22];
cx q[3], q[15];
rz(5.755821796768547) q[0];
rz(1.499236342405725) q[24];
cx q[4], q[26];
rz(0.015774329235065415) q[14];
rz(1.5106807688581563) q[19];
rz(5.896396853291438) q[7];
rz(4.803290886234669) q[25];
rz(0.8994017460247387) q[15];
rz(1.0586863549149839) q[26];
rz(4.315294708275801) q[17];
rz(4.391985159739418) q[21];
cx q[12], q[3];
rz(5.549558736087983) q[5];
rz(4.011148745523145) q[2];
rz(6.129677200842818) q[14];
rz(0.123005345621377) q[22];
rz(0.7237999190266654) q[18];
rz(4.794219092017203) q[0];
cx q[24], q[9];
rz(5.2662490279818615) q[16];
rz(2.0485550396992958) q[8];
rz(5.769897716582122) q[11];
cx q[23], q[13];
rz(5.619758460148776) q[10];
rz(6.267668863601868) q[4];
cx q[6], q[1];
cx q[19], q[7];
rz(5.6136907618121565) q[20];
rz(5.630407603870314) q[25];
rz(0.743271795363042) q[0];
rz(0.21900610671661774) q[14];
rz(2.6633460089407994) q[13];
rz(4.597275018388943) q[1];
cx q[20], q[15];
cx q[2], q[18];
cx q[26], q[21];
rz(0.9545036549183351) q[19];
cx q[17], q[11];
rz(1.6995375039682754) q[6];
rz(6.046084446177913) q[23];
rz(0.9146089824162219) q[7];
rz(2.5062621071894498) q[9];
rz(4.147821790686429) q[12];
rz(0.3490825505220476) q[16];
cx q[8], q[22];
rz(2.2307465816369465) q[10];
rz(2.2259464528462582) q[3];
cx q[5], q[4];
rz(0.6718552481535377) q[24];
rz(2.246876299578445) q[16];
rz(1.453677235721455) q[26];
rz(5.9824733250951425) q[15];
rz(0.6391589949820428) q[2];
rz(2.867261995943441) q[23];
rz(6.274869629147924) q[18];
rz(1.8819444747318226) q[7];
cx q[6], q[24];
rz(4.866988195669458) q[8];
cx q[9], q[10];
rz(1.6226203298916113) q[22];
rz(1.5412097611814226) q[20];
rz(1.555253078784917) q[21];
rz(0.038874886815880284) q[13];
cx q[14], q[11];
cx q[17], q[5];
rz(1.6946688004062278) q[25];
rz(1.9595285494340249) q[19];
rz(4.076775814217073) q[3];
rz(1.4849293880396142) q[4];
cx q[12], q[1];
rz(0.0038472966667272034) q[0];
rz(3.7190777779873865) q[13];
rz(5.7160875978243215) q[9];
rz(5.319638200834952) q[4];
rz(0.17783791198903712) q[2];
rz(1.5506248415098947) q[14];
rz(3.185282414930715) q[15];
rz(5.174574382037133) q[0];
rz(4.611661642240227) q[18];
rz(4.292508780456308) q[6];
rz(3.0812901280741563) q[10];
rz(2.6434084221588225) q[20];
cx q[23], q[1];
rz(3.77695420830799) q[12];
rz(5.362451237269248) q[25];
rz(4.56584791421215) q[17];
cx q[16], q[7];
rz(0.7901393613763927) q[3];
cx q[19], q[5];
rz(2.064427740792586) q[11];
rz(4.147349184872166) q[22];
rz(0.7278461720551364) q[8];
rz(3.5507565198878597) q[24];
cx q[21], q[26];
rz(0.012444105622706186) q[2];
rz(2.3219297173798914) q[26];
rz(5.431772445467775) q[9];
cx q[14], q[8];
rz(1.030398575589712) q[25];
rz(5.343235919413376) q[24];
rz(2.9009716173178495) q[23];
cx q[20], q[4];
rz(6.053496782477051) q[11];
rz(2.164473535985126) q[17];
rz(1.6799349231324048) q[12];
cx q[21], q[6];
rz(1.1684359935277127) q[7];
rz(1.117770623096323) q[5];
rz(5.525600721199385) q[22];
cx q[15], q[19];
rz(1.0658232135209231) q[0];
rz(4.611631587893012) q[3];
cx q[18], q[10];
rz(2.4157195966763942) q[13];
rz(1.9714204340717747) q[1];
rz(4.425416554517069) q[16];
rz(3.9308626320109488) q[0];
rz(6.2446370791517385) q[26];
cx q[1], q[18];
rz(1.885887753660469) q[4];
cx q[15], q[14];
rz(2.5286774202036617) q[16];
cx q[5], q[22];
rz(2.0211149648020754) q[19];
cx q[9], q[10];
rz(0.7907605867269121) q[25];
rz(1.3888979729921087) q[21];
rz(5.388935112882673) q[6];
rz(0.6241988388178605) q[12];
rz(3.7684580177727773) q[3];
rz(2.8867044197584524) q[7];
rz(5.151289166865409) q[8];
rz(4.8260096831326456) q[2];
rz(2.9599535504848586) q[20];
rz(2.429098497583968) q[17];
rz(1.5530857618250284) q[13];
cx q[11], q[23];
rz(4.833560915265166) q[24];
rz(1.299970747875047) q[12];
cx q[8], q[10];
rz(3.388100634599642) q[17];
rz(0.6080501319728309) q[18];
cx q[14], q[2];
rz(1.9070827265348407) q[21];
rz(5.006960905867374) q[13];
rz(4.406439541054745) q[20];
cx q[1], q[0];
cx q[15], q[25];
rz(3.1081187642099715) q[9];
rz(1.9485426217302806) q[24];
rz(0.38860012823943674) q[19];
rz(4.048586320476703) q[6];
rz(5.692266208119594) q[26];
cx q[5], q[16];
rz(4.6245982653128594) q[3];
cx q[11], q[7];
cx q[22], q[4];
rz(5.194723367365714) q[23];
rz(5.0027763888080985) q[7];
rz(3.801882539838321) q[4];
cx q[19], q[26];
cx q[0], q[3];
cx q[25], q[20];
cx q[8], q[9];
rz(1.4472516623499374) q[2];
rz(1.1333164911559137) q[24];
rz(1.3775943893613853) q[22];
rz(6.2571811870290075) q[11];
rz(0.9886176140830018) q[5];
rz(4.169518875482224) q[10];
rz(0.41555332606042783) q[6];
cx q[1], q[12];
cx q[18], q[17];
cx q[21], q[14];
rz(4.900742216472629) q[13];
rz(3.9690041504045817) q[23];
rz(2.633400312730859) q[16];
rz(4.509876723476786) q[15];
rz(4.93152000322996) q[22];
rz(1.2143756777375614) q[2];
rz(3.7759739415256797) q[26];
rz(5.977696436632081) q[25];
rz(1.6117675296602287) q[5];
rz(4.771149541291797) q[14];
rz(5.769485472330591) q[18];
rz(2.8733751160250076) q[10];
cx q[16], q[7];
cx q[4], q[24];
cx q[9], q[0];
cx q[19], q[13];
cx q[1], q[12];
rz(6.2825393610724625) q[6];
rz(2.9880150733225763) q[15];
rz(1.3420741081663954) q[11];
cx q[20], q[8];
rz(2.8403261323684004) q[23];
rz(4.4098554892514015) q[21];
rz(3.6188561622053896) q[3];
rz(0.9998502058783623) q[17];
rz(5.8110747715918505) q[17];
cx q[7], q[9];
rz(5.2874413098012765) q[21];
rz(3.023573753339901) q[19];
rz(0.2752996353521927) q[4];
rz(3.2136630152037333) q[11];
rz(5.731476387966325) q[24];
rz(5.173574347950098) q[2];
rz(3.9563200235339964) q[12];
rz(5.057429364210385) q[5];
rz(5.610306489755561) q[1];
rz(1.0630821991205908) q[3];
rz(3.252475441934622) q[18];
rz(1.3539759301411842) q[0];
rz(4.878417223337628) q[14];
rz(3.9914305024738845) q[26];
rz(2.449458225926237) q[23];
rz(4.094414034920475) q[16];
rz(1.3236131790114058) q[10];
cx q[15], q[22];
rz(1.3942600312798876) q[13];
rz(1.2932415623856504) q[25];
rz(4.308047920707526) q[6];
rz(3.962924955045168) q[20];
rz(2.455083384911126) q[8];
rz(4.512941715406904) q[11];
rz(2.248241100654724) q[16];
rz(3.976939767328115) q[26];
rz(3.6176872861079437) q[19];
rz(2.3184712217171777) q[4];
rz(0.7032711949259406) q[15];
rz(5.44229513935502) q[24];
rz(4.300080676628689) q[5];
rz(4.371378059399696) q[17];
rz(2.7194747931990855) q[13];
rz(5.292445828611087) q[18];
rz(4.627079107107717) q[1];
rz(6.21219734095799) q[7];
cx q[20], q[8];
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
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
measure q[20] -> c[20];
measure q[21] -> c[21];
measure q[22] -> c[22];
measure q[23] -> c[23];
measure q[24] -> c[24];
measure q[25] -> c[25];
measure q[26] -> c[26];