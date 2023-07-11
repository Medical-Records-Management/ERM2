import {Link} from "@remix-run/react"
import Slider from "~/Components/slider";
import Pricing from "~/Components/pricing";
import Features from "~/Components/features";
import Support from "~/Components/support";
import NavBar from "~/Components/navbar";
import Team from "~/Components/team";

export default function Index() {
  return (
    <div>
       {/* <Example></Example> */}
        <NavBar />
        <Slider />
        <Features />
        <Pricing />
        <Support />
        <Team />
    </div>
  );
}

 {/* <div id='pricing' className='relative grid gap-x-2 h-[100vh] p-5 md:grid-cols-3 place-items-center'>
    <Pricing title="Basic" img='./images/pricing-starter.png' price='10' color='green'/>
    <Pricing title="Standard" img='./images/pricing-business.png' price='20' color='orange'/>
    <Pricing title="Premium" img='./images/pricing-ultimate.png' price='35' color='purple'/>
  </div> */
}