OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c0[12];
u3(1.68651169800505,-0.716094371473713,1.41742248258121) q[6];
u3(2.24446638324182,-1.98467015004896,-2.15322697876123) q[9];
cx q[9],q[6];
u1(3.19733457158659) q[6];
u3(-1.51676476270545,0.0,0.0) q[9];
cx q[6],q[9];
u3(2.63276065163732,0.0,0.0) q[9];
cx q[9],q[6];
u3(2.74592548361927,2.50193069243410,0.508061102118735) q[6];
u3(1.59819878362513,0.432127427016519,-2.92413866086106) q[9];
u3(1.89222380440753,0.475533239599612,1.70638667122077) q[8];
u3(1.64096898768038,-2.74240161095443,-2.45648377129374) q[4];
cx q[4],q[8];
u1(0.508852550938624) q[8];
u3(-1.15103792135776,0.0,0.0) q[4];
cx q[8],q[4];
u3(1.80277683725961,0.0,0.0) q[4];
cx q[4],q[8];
u3(1.24120048870501,3.78453975927654,-1.47960886549445) q[8];
u3(0.556509195718177,1.67975112503431,-4.06116426464072) q[4];
u3(0.235959784642862,3.11519817287845,-2.26918519181026) q[3];
u3(0.532379160584293,-2.57890203933910,0.976315146826974) q[11];
cx q[11],q[3];
u1(1.04533901899330) q[3];
u3(-1.64655138142043,0.0,0.0) q[11];
cx q[3],q[11];
u3(-0.553183224073799,0.0,0.0) q[11];
cx q[11],q[3];
u3(0.412815840083663,0.353936660591259,1.25569671906390) q[3];
u3(1.22915427355071,-2.67610357959303,-3.12877537891558) q[11];
u3(2.79179014885198,-2.19375621567122,-0.316457921831245) q[5];
u3(1.69220179399914,-4.09189905459669,-0.533487353189521) q[10];
cx q[10],q[5];
u1(1.77684554543869) q[5];
u3(0.178312974198235,0.0,0.0) q[10];
cx q[5],q[10];
u3(0.612562624932134,0.0,0.0) q[10];
cx q[10],q[5];
u3(2.34725193710719,-2.41269375560730,0.450619478181894) q[5];
u3(0.485281402937688,1.06976063871668,-2.37967802391326) q[10];
u3(1.45396949099669,3.97956436910092,-1.30131768729499) q[0];
u3(0.849939972659445,1.51248420796185,-0.818513277292617) q[2];
cx q[2],q[0];
u1(-0.300263345762981) q[0];
u3(-1.52640862459236,0.0,0.0) q[2];
cx q[0],q[2];
u3(1.80852302038182,0.0,0.0) q[2];
cx q[2],q[0];
u3(1.76072866896843,-2.23765927502360,1.66815352335050) q[0];
u3(1.56333094683735,-0.970233637256424,-4.98484720799892) q[2];
u3(0.480068049244194,-1.46065055025520,0.792667455236130) q[7];
u3(0.202149214186702,0.193769452202299,-2.18676164064240) q[1];
cx q[1],q[7];
u1(1.25719958221299) q[7];
u3(-0.0777751046764050,0.0,0.0) q[1];
cx q[7],q[1];
u3(2.17111621023650,0.0,0.0) q[1];
cx q[1],q[7];
u3(2.19134650487252,2.04466963432863,0.108046772925731) q[7];
u3(0.913157332564323,0.354459724488490,5.51575677874033) q[1];
u3(1.50045297950620,2.31719639967056,-2.56066006391639) q[10];
u3(1.14085637295370,3.32976061393371,-2.35065341794936) q[7];
cx q[7],q[10];
u1(-0.128320241637785) q[10];
u3(0.673074326110362,0.0,0.0) q[7];
cx q[10],q[7];
u3(3.42330055433832,0.0,0.0) q[7];
cx q[7],q[10];
u3(2.04896871454781,3.20917942143010,-1.78087366224765) q[10];
u3(2.12537816967384,-0.412957230492225,-5.76181629501217) q[7];
u3(2.42297506304837,-0.298896668577576,0.626624625125958) q[2];
u3(2.29788512053328,-0.815784025895471,-1.85997468313509) q[8];
cx q[8],q[2];
u1(1.62573314989169) q[2];
u3(-3.23904225478628,0.0,0.0) q[8];
cx q[2],q[8];
u3(2.57096598007649,0.0,0.0) q[8];
cx q[8],q[2];
u3(3.04547236151555,2.96639702078949,-1.51179337317425) q[2];
u3(1.24090867846989,4.17764633640708,1.80286575969784) q[8];
u3(2.46964764475586,3.48305271456267,-0.369526986031139) q[11];
u3(1.59850862520140,3.48037568711344,0.558819631635112) q[4];
cx q[4],q[11];
u1(1.36786376720251) q[11];
u3(-3.54676670253932,0.0,0.0) q[4];
cx q[11],q[4];
u3(2.43182469176759,0.0,0.0) q[4];
cx q[4],q[11];
u3(2.12437391140637,-2.98321813233164,1.55773643539832) q[11];
u3(3.05378795404923,3.00160386235642,-1.45253548185418) q[4];
u3(0.896189050472342,3.22918770371903,-2.21084748932263) q[6];
u3(0.713046309037842,1.54727036906374,-2.14656106127054) q[3];
cx q[3],q[6];
u1(2.47403819408887) q[6];
u3(-2.01753748486825,0.0,0.0) q[3];
cx q[6],q[3];
u3(3.25306883210614,0.0,0.0) q[3];
cx q[3],q[6];
u3(2.51283612765064,0.210069748017253,1.64535003778123) q[6];
u3(1.35969039982866,-0.183833210248246,-5.78268764493807) q[3];
u3(1.75987145140493,3.59077730268799,-0.604322583125137) q[5];
u3(1.95945238762591,2.84141849565184,-1.30578044508998) q[0];
cx q[0],q[5];
u1(1.98905445591009) q[5];
u3(-2.89223654377159,0.0,0.0) q[0];
cx q[5],q[0];
u3(0.832506006052138,0.0,0.0) q[0];
cx q[0],q[5];
u3(1.09950274904550,-2.99151415342051,0.0802331003110281) q[5];
u3(2.71914679539416,-2.97655717265437,-0.992935381738077) q[0];
u3(2.84280451209735,0.994989309925983,-3.08773501049755) q[1];
u3(2.94708441412461,-0.0183810778737641,-4.56094336818150) q[9];
cx q[9],q[1];
u1(3.29073502024572) q[1];
u3(-4.23999585796830,0.0,0.0) q[9];
cx q[1],q[9];
u3(-0.324657691005818,0.0,0.0) q[9];
cx q[9],q[1];
u3(1.76971011934970,-0.559979505519561,-1.08176590135389) q[1];
u3(0.976934975086052,3.00489471539095,2.97220492572464) q[9];
u3(2.22223172951523,-0.245950324335028,2.42825286201042) q[5];
u3(2.26046327161950,-3.65379650364752,-2.19943499164890) q[8];
cx q[8],q[5];
u1(2.64284533627021) q[5];
u3(-1.82474744222266,0.0,0.0) q[8];
cx q[5],q[8];
u3(3.29432301263153,0.0,0.0) q[8];
cx q[8],q[5];
u3(1.34680977177593,2.24535314295291,-0.223545457753646) q[5];
u3(1.84149393500141,-0.153022011044791,1.19478820749840) q[8];
u3(0.853839568290930,0.693123894486357,-1.08068463097614) q[7];
u3(0.648651272709943,-0.964047746843289,-0.352835741897014) q[9];
cx q[9],q[7];
u1(1.16866672289275) q[7];
u3(-3.18362597968414,0.0,0.0) q[9];
cx q[7],q[9];
u3(0.757858890388685,0.0,0.0) q[9];
cx q[9],q[7];
u3(0.725272271554638,1.09825848117227,1.50233894945690) q[7];
u3(0.320001234002285,2.72182215277190,2.66238217124031) q[9];
u3(1.47202446685604,2.09204460065113,-0.307071627733680) q[4];
u3(2.44197890463342,0.269222498484944,-2.93970601747417) q[6];
cx q[6],q[4];
u1(0.133765355509468) q[4];
u3(-0.648103598397905,0.0,0.0) q[6];
cx q[4],q[6];
u3(1.73722803538175,0.0,0.0) q[6];
cx q[6],q[4];
u3(1.78352083626164,-2.09477735259274,2.38149767870823) q[4];
u3(0.307402616975535,1.17476487642802,3.71481702134707) q[6];
u3(1.98973752483879,0.0285419685549324,-0.404481389660520) q[0];
u3(1.53496829506791,1.02624885753372,-4.20916598742817) q[10];
cx q[10],q[0];
u1(-0.225610385549295) q[0];
u3(-1.72227355460564,0.0,0.0) q[10];
cx q[0],q[10];
u3(0.653108550905720,0.0,0.0) q[10];
cx q[10],q[0];
u3(2.17149105736333,1.50735683587703,-1.46655686641656) q[0];
u3(0.689407443865838,-5.00653598518219,-0.474592679946937) q[10];
u3(2.62398453064974,0.860567222831363,1.33248871821625) q[11];
u3(1.29536028866987,-5.68860312573626,0.565028878407098) q[1];
cx q[1],q[11];
u1(-0.0463832530618746) q[11];
u3(1.15924737535366,0.0,0.0) q[1];
cx q[11],q[1];
u3(3.51567369542100,0.0,0.0) q[1];
cx q[1],q[11];
u3(0.380237489787912,1.43414263269503,0.848320503865723) q[11];
u3(2.67934012739309,0.551882318487698,-3.35926415543971) q[1];
u3(1.52230712870830,1.30803230469210,-3.22366689316008) q[3];
u3(0.573365597791560,2.06209050608023,-2.17883195979747) q[2];
cx q[2],q[3];
u1(1.43792249854120) q[3];
u3(-2.72926778635507,0.0,0.0) q[2];
cx q[3],q[2];
u3(3.02064295329964,0.0,0.0) q[2];
cx q[2],q[3];
u3(1.43989898261963,-1.85139668541500,-0.927285015966830) q[3];
u3(1.03428072379309,5.44150843805990,0.258739376380532) q[2];
u3(2.41680024203310,-0.470997190767594,1.73010898222113) q[11];
u3(2.15077244408272,-1.61378850463215,-1.23073659759425) q[7];
cx q[7],q[11];
u1(0.446768335020559) q[11];
u3(-1.69481327558889,0.0,0.0) q[7];
cx q[11],q[7];
u3(2.75502149286239,0.0,0.0) q[7];
cx q[7],q[11];
u3(1.05933131614054,-1.12794411812534,2.79130441195347) q[11];
u3(1.95784938238073,-1.46263969732678,3.01546794922404) q[7];
u3(2.43035640197178,0.526141193216919,1.81152870992160) q[3];
u3(1.22527104075713,-3.23493443852889,-2.49543380482700) q[1];
cx q[1],q[3];
u1(0.00511417847475149) q[3];
u3(-0.694724282416146,0.0,0.0) q[1];
cx q[3],q[1];
u3(1.89291878323729,0.0,0.0) q[1];
cx q[1],q[3];
u3(1.70558988253139,-3.03305444510929,1.95674150547038) q[3];
u3(1.77254764052342,-0.634084419684078,4.84728509079389) q[1];
u3(2.78925991806759,2.26105269974659,-0.197480442578587) q[0];
u3(1.80839897504756,1.04442137102291,-3.29051609231746) q[10];
cx q[10],q[0];
u1(-0.873714682168103) q[0];
u3(1.00747739357852,0.0,0.0) q[10];
cx q[0],q[10];
u3(3.61616824840543,0.0,0.0) q[10];
cx q[10],q[0];
u3(0.733523866055825,3.80303565239639,-0.892786120180290) q[0];
u3(1.24581418381927,3.66261054083054,2.51266987570271) q[10];
u3(1.82280788735542,1.07093043990139,2.07030271823951) q[8];
u3(2.10995472056642,-1.25116933649015,-1.18572055176898) q[2];
cx q[2],q[8];
u1(2.11122959938407) q[8];
u3(-2.92054062264765,0.0,0.0) q[2];
cx q[8],q[2];
u3(1.67355725476029,0.0,0.0) q[2];
cx q[2],q[8];
u3(0.978943510219272,0.596648365519163,-2.31375798675587) q[8];
u3(1.28126177133897,-1.43484188148805,4.55400032975648) q[2];
u3(1.35625318258903,2.65364803862521,-2.72809426378446) q[4];
u3(1.49182446809987,2.96444841814074,-3.16626664711706) q[5];
cx q[5],q[4];
u1(1.80825236162449) q[4];
u3(-3.43954026835111,0.0,0.0) q[5];
cx q[4],q[5];
u3(1.43602884911577,0.0,0.0) q[5];
cx q[5],q[4];
u3(1.62173364354464,-0.131386691397926,3.04641827451457) q[4];
u3(2.81482565287335,-0.441745057529094,3.99900984187629) q[5];
u3(1.45188585054184,-0.985469342592164,-0.818089976842319) q[9];
u3(1.59812521273460,-4.35805080718573,1.02830366859876) q[6];
cx q[6],q[9];
u1(1.01519492764310) q[9];
u3(-1.39843654243245,0.0,0.0) q[6];
cx q[9],q[6];
u3(-0.288131884304375,0.0,0.0) q[6];
cx q[6],q[9];
u3(1.07717849518319,-2.73596061887521,0.946047148932265) q[9];
u3(0.574329027170190,-5.38450509979489,0.324174742996429) q[6];
u3(0.912154555776174,2.16938126101034,-2.79292404035461) q[8];
u3(0.396011495396116,1.28597134482466,-2.64236581015378) q[6];
cx q[6],q[8];
u1(3.55625477458756) q[8];
u3(-1.80121047751989,0.0,0.0) q[6];
cx q[8],q[6];
u3(1.57216472316059,0.0,0.0) q[6];
cx q[6],q[8];
u3(2.56712708967818,-1.38469979758197,2.32754274046610) q[8];
u3(0.863150598886175,1.51839223569815,4.56504643039547) q[6];
u3(1.31156005519095,-0.485851559570952,1.89851165884624) q[9];
u3(1.18122324656090,-1.07039869751322,-2.43259530248752) q[10];
cx q[10],q[9];
u1(1.25336881264356) q[9];
u3(-2.65177491532786,0.0,0.0) q[10];
cx q[9],q[10];
u3(3.26053502542400,0.0,0.0) q[10];
cx q[10],q[9];
u3(0.319533631999930,1.45415374540135,-4.20905141550287) q[9];
u3(1.47672517458646,-0.263021354681519,3.70812892030704) q[10];
u3(2.14172756839604,0.717124959895980,-3.40859974987890) q[2];
u3(2.06885571745459,2.75647318604874,-2.98305221952708) q[3];
cx q[3],q[2];
u1(1.68664084191149) q[2];
u3(-2.61059612371569,0.0,0.0) q[3];
cx q[2],q[3];
u3(-0.0633912668909078,0.0,0.0) q[3];
cx q[3],q[2];
u3(2.59951404180607,1.36128975514536,0.827935346111285) q[2];
u3(1.32508086104712,0.249750777010165,0.272176837068894) q[3];
u3(0.444325123222747,2.56274355583983,-2.22078656038830) q[4];
u3(1.22130548519124,0.0882020515813919,-2.55124523842009) q[5];
cx q[5],q[4];
u1(1.53442244783723) q[4];
u3(-2.32830088244473,0.0,0.0) q[5];
cx q[4],q[5];
u3(3.76607820312171,0.0,0.0) q[5];
cx q[5],q[4];
u3(1.32277892490533,-1.08436516457562,1.86317379016730) q[4];
u3(1.80521414500267,5.33637883574266,0.853578470266521) q[5];
u3(1.79599184581557,0.845474401651582,-1.04190141784972) q[1];
u3(2.75756731104131,-4.94125669625073,0.190961826574424) q[11];
cx q[11],q[1];
u1(-0.571434075934750) q[1];
u3(-2.06893051730035,0.0,0.0) q[11];
cx q[1],q[11];
u3(1.73363629838117,0.0,0.0) q[11];
cx q[11],q[1];
u3(1.45592810960131,-0.322419350068896,-2.05713132496378) q[1];
u3(2.92827079511425,-4.09856686821279,-0.428142969800730) q[11];
u3(2.38913680829973,0.614078351459052,1.98630476483599) q[7];
u3(1.67735294448840,-2.78587101296481,-2.55384459247431) q[0];
cx q[0],q[7];
u1(3.76218392974475) q[7];
u3(-0.829855290293749,0.0,0.0) q[0];
cx q[7],q[0];
u3(1.56949163850734,0.0,0.0) q[0];
cx q[0],q[7];
u3(1.17537245625765,0.514247824592679,-1.60716074976980) q[7];
u3(1.38016261540700,0.892346056365917,-2.07881361952668) q[0];
u3(1.92085131954841,-0.855845935467192,-0.147328917938623) q[6];
u3(0.757077722805243,-4.02536429374816,-0.708168043717594) q[9];
cx q[9],q[6];
u1(2.10549072371789) q[6];
u3(-2.28071551136705,0.0,0.0) q[9];
cx q[6],q[9];
u3(-0.0840138065107738,0.0,0.0) q[9];
cx q[9],q[6];
u3(1.63262254556922,-3.65972265781114,1.94313938151120) q[6];
u3(0.940642176823148,-1.21903423683783,-0.400400382397736) q[9];
u3(0.375019271382324,3.18133937189791,-2.41261445926596) q[5];
u3(1.57774932060554,1.43292262470411,-1.87677238393647) q[11];
cx q[11],q[5];
u1(0.134643011516355) q[5];
u3(-1.55178262306010,0.0,0.0) q[11];
cx q[5],q[11];
u3(2.06312649702583,0.0,0.0) q[11];
cx q[11],q[5];
u3(2.60720556208809,-1.63660024938485,4.25122649803863) q[5];
u3(1.77552581074895,3.89715706492313,0.837016955887242) q[11];
u3(0.488476821672697,3.15769895239180,-2.38063349938379) q[7];
u3(1.44411173690174,1.53969996744474,-1.02516092872508) q[3];
cx q[3],q[7];
u1(4.19315076629392) q[7];
u3(-3.67671667922125,0.0,0.0) q[3];
cx q[7],q[3];
u3(-0.199215031731376,0.0,0.0) q[3];
cx q[3],q[7];
u3(2.04190282441724,1.99860537424438,-3.61640886082587) q[7];
u3(2.08082279142058,0.477633926435847,3.78014621462212) q[3];
u3(2.02344699972889,-0.0154839482469032,-0.706969510395238) q[2];
u3(1.26556969017626,0.445426071040097,-4.62693177884108) q[1];
cx q[1],q[2];
u1(1.19850885189379) q[2];
u3(-0.223333903919585,0.0,0.0) q[1];
cx q[2],q[1];
u3(2.66300564373100,0.0,0.0) q[1];
cx q[1],q[2];
u3(1.58630347316905,-1.06181415979658,-0.416388079569544) q[2];
u3(0.794853660016690,-0.392131866202153,3.72802845351427) q[1];
u3(0.756358813755573,-2.63952050934256,2.02318750157426) q[10];
u3(0.369297979947042,1.14641103425067,-3.59929353749460) q[0];
cx q[0],q[10];
u1(1.13787794116640) q[10];
u3(-0.677672082520614,0.0,0.0) q[0];
cx q[10],q[0];
u3(-0.202976942196395,0.0,0.0) q[0];
cx q[0],q[10];
u3(1.02494085350087,0.124942156556863,1.51127102880169) q[10];
u3(2.03932812742566,-5.34056060152748,-0.398949130444380) q[0];
u3(0.930610579658087,0.511933642936769,1.07125526294979) q[4];
u3(1.71475762938732,-2.74882128891178,-0.806508347634400) q[8];
cx q[8],q[4];
u1(3.42080380800082) q[4];
u3(-0.938947438362926,0.0,0.0) q[8];
cx q[4],q[8];
u3(1.73910404769243,0.0,0.0) q[8];
cx q[8],q[4];
u3(1.85065585440169,0.485977355358550,-3.24762109066554) q[4];
u3(2.36545986137831,5.72856617464105,-0.364164750792565) q[8];
u3(0.589452375611575,-2.69514377568617,1.28000590147799) q[1];
u3(0.714151718367994,0.909908525875736,-2.53015035687857) q[8];
cx q[8],q[1];
u1(3.19473147098002) q[1];
u3(-0.819905437379659,0.0,0.0) q[8];
cx q[1],q[8];
u3(1.48038673989758,0.0,0.0) q[8];
cx q[8],q[1];
u3(1.63950156105885,2.41294000476604,-0.0790362859888014) q[1];
u3(1.57749143810445,5.37803593413927,0.613506916949034) q[8];
u3(0.482947087054975,-2.19545335772845,0.917257244901475) q[6];
u3(0.338145514518992,-2.97561365066038,1.73442145718067) q[0];
cx q[0],q[6];
u1(0.581323698431498) q[6];
u3(-1.04194162960507,0.0,0.0) q[0];
cx q[6],q[0];
u3(1.66407783523200,0.0,0.0) q[0];
cx q[0],q[6];
u3(1.65002464808794,-0.515646975511430,3.06061687329440) q[6];
u3(1.82114928801122,3.80631453164641,1.88873056184172) q[0];
u3(2.07463102839254,-1.94806457697630,0.708195208915768) q[9];
u3(2.16439196624273,-4.49975237786953,-1.50192736379856) q[11];
cx q[11],q[9];
u1(3.32292951142753) q[9];
u3(-0.803759621738030,0.0,0.0) q[11];
cx q[9],q[11];
u3(1.88278371296965,0.0,0.0) q[11];
cx q[11],q[9];
u3(1.57175504801828,-4.16220351097649,1.30174600200959) q[9];
u3(1.01019018701513,-1.58495697023125,-1.11940632680125) q[11];
u3(1.84537103122941,0.228505673232111,0.934711286243235) q[4];
u3(0.213263819573806,-4.50903126086727,-0.350983812713269) q[2];
cx q[2],q[4];
u1(2.29419468843491) q[4];
u3(-2.71795136144510,0.0,0.0) q[2];
cx q[4],q[2];
u3(1.51655958104908,0.0,0.0) q[2];
cx q[2],q[4];
u3(1.08858113632168,1.36506788766417,-1.31820697274946) q[4];
u3(1.77317706941146,-2.45387522727924,-1.39317787003900) q[2];
u3(0.203912538825807,1.29520387392797,-0.576968077538691) q[5];
u3(1.40082308245105,-3.91559263634883,1.50424376365241) q[3];
cx q[3],q[5];
u1(1.60663605929391) q[5];
u3(-2.52896478423388,0.0,0.0) q[3];
cx q[5],q[3];
u3(0.386652048682512,0.0,0.0) q[3];
cx q[3],q[5];
u3(0.870918775466988,3.23604905151611,-2.32872556352503) q[5];
u3(2.47752635465178,-3.04143484178268,0.101015474659607) q[3];
u3(0.786826893704891,-2.78682304155376,2.41913347152408) q[10];
u3(0.551426837827588,-3.94536244745258,2.07322792251140) q[7];
cx q[7],q[10];
u1(0.884413066367903) q[10];
u3(-0.250199848859540,0.0,0.0) q[7];
cx q[10],q[7];
u3(1.79564189930070,0.0,0.0) q[7];
cx q[7],q[10];
u3(1.63413663388841,1.21928963880100,-2.87347130970579) q[10];
u3(1.12931703116022,-2.74473463794773,-2.91699766308502) q[7];
u3(0.944159989493706,3.09716734428968,-2.43692083741348) q[1];
u3(0.943169684032598,2.60671612093397,-2.18543264095787) q[4];
cx q[4],q[1];
u1(-0.176132428508595) q[1];
u3(-1.82318321155393,0.0,0.0) q[4];
cx q[1],q[4];
u3(0.583633886782064,0.0,0.0) q[4];
cx q[4],q[1];
u3(1.23943950085815,-2.79193061099248,2.87765940075007) q[1];
u3(1.75345269187136,-1.52670698973827,4.01321277114486) q[4];
u3(2.73256429475403,0.609570053480265,0.388074038118931) q[3];
u3(1.25709885562426,-3.90220224132998,-1.12498241635934) q[5];
cx q[5],q[3];
u1(3.09992470357340) q[3];
u3(-2.21862951895830,0.0,0.0) q[5];
cx q[3],q[5];
u3(1.41145916250941,0.0,0.0) q[5];
cx q[5],q[3];
u3(1.05758960819989,0.660925877822360,-0.477949375191159) q[3];
u3(1.54388387460070,-0.532690224303938,-1.05846192061741) q[5];
u3(1.03168680569262,-1.36350447126743,0.609180144114007) q[9];
u3(1.72394714354753,-4.22257335861237,-0.216450245527234) q[8];
cx q[8],q[9];
u1(0.995680964831864) q[9];
u3(-0.134823842348876,0.0,0.0) q[8];
cx q[9],q[8];
u3(2.35485997092122,0.0,0.0) q[8];
cx q[8],q[9];
u3(0.854716753151681,1.01363330024658,-0.863901562143666) q[9];
u3(1.05639416411527,-1.15460643083436,-1.81044734361296) q[8];
u3(2.17235948048332,-0.00161883011439945,1.58805036122827) q[11];
u3(2.12034252366275,-1.08020481495213,-1.50312295246765) q[0];
cx q[0],q[11];
u1(0.447141331819449) q[11];
u3(-3.28888396376550,0.0,0.0) q[0];
cx q[11],q[0];
u3(1.41073181142842,0.0,0.0) q[0];
cx q[0],q[11];
u3(2.35777642224605,0.402340973327233,2.09360462477639) q[11];
u3(1.27464272885883,-2.70102828945775,-0.571820802699682) q[0];
u3(0.914835810757443,0.430073059944475,1.74058982732460) q[6];
u3(2.04869082237103,-0.558943416812190,-2.98424485848852) q[2];
cx q[2],q[6];
u1(0.0285608204786805) q[6];
u3(-2.13884649140584,0.0,0.0) q[2];
cx q[6],q[2];
u3(0.840983159824467,0.0,0.0) q[2];
cx q[2],q[6];
u3(1.83260005893428,-0.589381519574327,-1.26127773989321) q[6];
u3(1.34503485710470,-1.04022231433718,2.57602074019148) q[2];
u3(1.94994901248295,2.31696044981832,-1.79508223224552) q[7];
u3(1.02609396886791,1.20196767641344,-0.283257108529705) q[10];
cx q[10],q[7];
u1(1.19754673822640) q[7];
u3(-0.606808131424486,0.0,0.0) q[10];
cx q[7],q[10];
u3(2.85571224987444,0.0,0.0) q[10];
cx q[10],q[7];
u3(1.36470141896567,-2.98825243388626,1.75583821982123) q[7];
u3(1.24944274938877,1.98760024007062,0.572453828922626) q[10];
u3(1.57594563171834,-0.960927034605646,-0.827667986102628) q[7];
u3(1.05840202617253,-2.50893132357817,0.100390214300783) q[3];
cx q[3],q[7];
u1(0.295452488075130) q[7];
u3(-1.61972803615897,0.0,0.0) q[3];
cx q[7],q[3];
u3(3.02935657097571,0.0,0.0) q[3];
cx q[3],q[7];
u3(0.765179357009629,-4.19576711595244,1.54882125898537) q[7];
u3(1.90631233722885,-3.60836979889394,-1.49892623938252) q[3];
u3(0.587629878337376,-3.19668212460156,2.35243758019761) q[0];
u3(0.900328322412500,-4.07056066149505,2.04791550815965) q[9];
cx q[9],q[0];
u1(0.953057491531142) q[0];
u3(-1.38547048717486,0.0,0.0) q[9];
cx q[0],q[9];
u3(-0.199714176506415,0.0,0.0) q[9];
cx q[9],q[0];
u3(1.61975833812011,-0.668786081572092,-2.25840729274550) q[0];
u3(0.668875757529149,-5.17377274873059,-0.244265445022727) q[9];
u3(2.02530191345653,2.48828145238613,-2.14422920987549) q[10];
u3(1.04293348309518,1.49614190648994,-2.54492657434168) q[2];
cx q[2],q[10];
u1(2.54738475344576) q[10];
u3(-2.13552621753656,0.0,0.0) q[2];
cx q[10],q[2];
u3(1.36779953035210,0.0,0.0) q[2];
cx q[2],q[10];
u3(2.23401621783696,0.156619348981170,2.39976139136048) q[10];
u3(0.748584570222739,-0.348343434074001,5.74565059257407) q[2];
u3(1.86652919123484,-3.15968817262797,2.28439660918057) q[8];
u3(0.863186168444786,2.85682982784336,-1.00965873125185) q[5];
cx q[5],q[8];
u1(1.55222506038905) q[8];
u3(-2.89190257035410,0.0,0.0) q[5];
cx q[8],q[5];
u3(0.998338279873914,0.0,0.0) q[5];
cx q[5],q[8];
u3(2.14530268598026,1.02622995196416,0.883945118786554) q[8];
u3(1.30693437248649,-0.273424888612895,4.33554448374858) q[5];
u3(1.64905813488404,1.06289086708594,1.83967631872490) q[6];
u3(1.67564873770366,-2.02690617313496,-2.06995510542307) q[11];
cx q[11],q[6];
u1(0.444874428320581) q[6];
u3(-1.47640546443276,0.0,0.0) q[11];
cx q[6],q[11];
u3(-0.185587615342028,0.0,0.0) q[11];
cx q[11],q[6];
u3(1.13049864751358,1.42015294086418,-2.25931732340909) q[6];
u3(1.59517368627609,-0.788854267781443,-2.93224544542982) q[11];
u3(2.06961656014856,1.63494931419134,-0.636521237714339) q[4];
u3(2.19531973736684,0.310945836321891,-3.38700701977398) q[1];
cx q[1],q[4];
u1(0.956793169197966) q[4];
u3(-0.784852821387655,0.0,0.0) q[1];
cx q[4],q[1];
u3(2.64032518416763,0.0,0.0) q[1];
cx q[1],q[4];
u3(0.746909056192582,-1.48761338035302,3.24360129301024) q[4];
u3(1.43510890323978,5.16120809373574,-0.439261426492645) q[1];
u3(0.562253754496276,0.177949947898835,0.0865890105365931) q[10];
u3(0.185311444023484,-1.95113889469080,-0.638253587974516) q[9];
cx q[9],q[10];
u1(2.99956976626317) q[10];
u3(-0.950631506571144,0.0,0.0) q[9];
cx q[10],q[9];
u3(1.61600909547791,0.0,0.0) q[9];
cx q[9],q[10];
u3(1.16156842463171,-2.84982753165885,3.35477195649605) q[10];
u3(1.36801872527168,-3.64007759952046,-1.18781769496470) q[9];
u3(1.02737463338105,1.65276597918585,-3.85513182225629) q[5];
u3(2.02156278296645,3.11537011163462,-2.48300893713303) q[0];
cx q[0],q[5];
u1(1.84565263243169) q[5];
u3(-0.678039351153075,0.0,0.0) q[0];
cx q[5],q[0];
u3(2.98624401809710,0.0,0.0) q[0];
cx q[0],q[5];
u3(0.949237929921411,1.38687845460748,-0.609175478308661) q[5];
u3(1.60173070470851,0.691063342411047,1.32044495924982) q[0];
u3(2.56498442650376,-2.86139805892747,3.26340998039706) q[6];
u3(0.972697855148455,3.36652874634585,-2.45669189935698) q[3];
cx q[3],q[6];
u1(0.322744703332132) q[6];
u3(-1.14005642599799,0.0,0.0) q[3];
cx q[6],q[3];
u3(1.68339302595969,0.0,0.0) q[3];
cx q[3],q[6];
u3(1.50398680404485,3.81864371887366,-1.58734692689237) q[6];
u3(0.110214885822286,-2.41794887273799,0.487503977500873) q[3];
u3(1.01805525163722,0.897721226255501,-0.872979788205809) q[7];
u3(0.598402081841939,-2.77920543743980,0.941590248743072) q[8];
cx q[8],q[7];
u1(1.15868906003835) q[7];
u3(-0.0898002048043627,0.0,0.0) q[8];
cx q[7],q[8];
u3(1.64202475734013,0.0,0.0) q[8];
cx q[8],q[7];
u3(1.91843750821429,-0.105624670285008,2.18144873278510) q[7];
u3(0.339693944912319,2.11747860993412,3.69914078076246) q[8];
u3(1.30770672778659,1.53771793937713,-3.65259040965333) q[11];
u3(0.998700653581647,-1.96693903888202,2.73499639585157) q[4];
cx q[4],q[11];
u1(-0.0173418565826284) q[11];
u3(-1.50610672302870,0.0,0.0) q[4];
cx q[11],q[4];
u3(2.21504492578840,0.0,0.0) q[4];
cx q[4],q[11];
u3(0.994331710979958,-0.172775576500717,2.94482294900786) q[11];
u3(0.192812228793557,4.70998141384139,-0.358587677455677) q[4];
u3(0.790958872584938,-2.29757375187169,3.15095511339440) q[1];
u3(0.623069290618167,1.95884120157957,-3.87061524579403) q[2];
cx q[2],q[1];
u1(0.907856892013079) q[1];
u3(-3.16732784803737,0.0,0.0) q[2];
cx q[1],q[2];
u3(2.12225523397580,0.0,0.0) q[2];
cx q[2],q[1];
u3(1.29821843886442,0.723900601569865,-0.865467769984330) q[1];
u3(2.48207880626292,0.705874901087888,-5.23181127459597) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11];
measure q[0] -> c0[0];
measure q[1] -> c0[1];
measure q[2] -> c0[2];
measure q[3] -> c0[3];
measure q[4] -> c0[4];
measure q[5] -> c0[5];
measure q[6] -> c0[6];
measure q[7] -> c0[7];
measure q[8] -> c0[8];
measure q[9] -> c0[9];
measure q[10] -> c0[10];
measure q[11] -> c0[11];
