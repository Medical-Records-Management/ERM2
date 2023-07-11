// import { NavLink } from "@remix-run/react";

// export default function Navbar()
// {
//     return(
//       <> 
//         <nav className="lg:flex justify-center items-center text-white mx-auto w-full py-2 px-8 md:hidden sm:hidden">
//             <div className="lg:w-full bg-blue-gray-300 p-4">
//               <span className="text-2xl font-bold text-[#183BF0]">Octaven ERM.</span>
//             </div>
//             <div className="bg-red-500">
//                <ul className="flex">
//                 <li className="p-4 hover:text-[#183BF0]"><NavLink to="#about">About</NavLink></li>
//                 <li className="p-4 hover:text-[#183BF0]"><NavLink to="#pricing">Pricing</NavLink></li>
//                 <li className="p-4 hover:text-[#183BF0]"><NavLink to="#features">Features</NavLink></li>
//               </ul>
//             </div>
//             <div className="bg-green-400 flex p-4 place-items-right">
//                 <NavLink to="#" className="mx-4"><button className="bg-[#183BF0] rounded w-20 h-auto">Sign-in</button></NavLink>
//                 <NavLink to="#" className="mx-4"><button className="bg-[#183BF0] rounded w-20 h-auto">Sign-up</button></NavLink>
//             </div>
//         </nav>
//         <nav className="sm:flex md:flex lg:hidden">
//             <div className="w-2/4  bg-blue-gray-300 p-4">
//               <span className="text-2xl font-bold text-[#183BF0]">ERM.</span>
//             </div>
//             <div className="ml-8">
//                 <ul className="flex mx-4">
//                   <li className="p-4">A</li>
//                   <li className="p-4">B</li>
//                   <li className="p-4">C</li>
//                 </ul>
//             </div>
//         </nav>
//       </> 

//     );
// }

// import React from "react";
// import { NavLink } from "@remix-run/react";
// import {
//   Navbar,
//   MobileNav,
//   Typography,
//   Button,
//   IconButton,
// } from "@material-tailwind/react";
 
// export default function NavBar() {
//   const [openNav, setOpenNav] = React.useState(false);
 
//   React.useEffect(() => {
//     window.addEventListener(
//       "resize",
//       () => window.innerWidth >= 960 && setOpenNav(false)
//     );
//   }, []);
 
//   const navList = (
//     <ul className="mb-4 mt-2 flex flex-col gap-2 lg:mb-0 lg:mt-0 lg:flex-row lg:items-center lg:gap-6">
//       <Typography
//         as="li"
//         variant="small"
//         color="blue-gray"
//         className="p-1 font-normal"
//       >
//         <NavLink to="#about" className="flex items-center">
//           About
//         </NavLink>
//       </Typography>
//       <Typography
//         as="li"
//         variant="small"
//         color="blue-gray"
//         className="p-1 font-normal"
//       >
//         <NavLink to="#features" className="flex items-center">
//           Features
//         </NavLink>
//       </Typography>
//       <Typography
//         as="li"
//         variant="small"
//         color="blue-gray"
//         className="p-1 font-normal"
//       >
//         <NavLink to="#pricing" className="flex items-center">
//           Pricing
//         </NavLink>
//       </Typography>
//       <Typography
//         as="li"
//         variant="small"
//         color="blue-gray"
//         className="p-1 font-normal"
//       >
//         <a href="#" className="flex items-center">
//           Docs
//         </a>
//       </Typography>
//     </ul>
//   );
 
//   return (
//     <>
//       <Navbar className="sticky top z-10 h-max max-w-full rounded-none py-2 px-4 lg:px-8 lg:py-4">
//         <div className="flex items-center justify-between text-blue-gray-900">
//           <Typography
//             as="a"
//             href="/"
//             className="mr-4 cursor-pointer py-1.5 font-medium"
//           >
//             MRM
//           </Typography>
//           <div className="flex items-center gap-4">
//             <div className="mr-4 hidden lg:block">{navList}</div>
//             <Button
//               variant="gradient"
//               size="sm"
//               className="hidden lg:inline-block"
//             >
//               <span>Account</span>
//             </Button>
//             <IconButton
//               variant="text"
//               className="ml-auto h-6 w-6 text-inherit hover:bg-transparent focus:bg-transparent active:bg-transparent lg:hidden"
//               ripple={false}
//               onClick={() => setOpenNav(!openNav)}
//             >
//               {openNav ? (
//                 <svg
//                   xmlns="http://www.w3.org/2000/svg"
//                   fill="none"
//                   className="h-6 w-6"
//                   viewBox="0 0 24 24"
//                   stroke="currentColor"
//                   strokeWidth={2}
//                 >
//                   <path
//                     strokeLinecap="round"
//                     strokeLinejoin="round"
//                     d="M6 18L18 6M6 6l12 12"
//                   />
//                 </svg>
//               ) : (
//                 <svg
//                   xmlns="http://www.w3.org/2000/svg"
//                   className="h-6 w-6"
//                   fill="none"
//                   stroke="currentColor"
//                   strokeWidth={2}
//                 >
//                   <path
//                     strokeLinecap="round"
//                     strokeLinejoin="round"
//                     d="M4 6h16M4 12h16M4 18h16"
//                   />
//                 </svg>
//               )}
//             </IconButton>
//           </div>
//         </div>
//         <MobileNav open={openNav}>
//           {navList}
//           <Button variant="gradient" size="sm" fullWidth className="mb-2">
//             <span>Buy Now</span>
//           </Button>
//         </MobileNav>
//       </Navbar>
//     </>
//   );
// }

import { NavLink } from "@remix-run/react";
import React from "react";
import {
  Navbar,
  Collapse,
  Typography,
  IconButton,
} from "@material-tailwind/react";
import { 
  Bars3Icon,
  XMarkIcon,
  UserCircleIcon,
  InformationCircleIcon,
  Square3Stack3DIcon,
  CreditCardIcon,
  PhoneIcon
} from "@heroicons/react/24/outline";
 
function NavList() {
  return (
    <ul className="my-2 text-black flex flex-col gap-2 lg:mb-0 lg:mt-0 lg:flex-row lg:items-center lg:gap-6">
      <Typography
        as="li"
        variant="small"
        color="blue-gray"
        className="p-1 font-medium"
      >
        <NavLink to="#about" className="flex items-center hover:text-blue-500 transition-colors">
          <InformationCircleIcon className="w-4 h-4 mr-1"></InformationCircleIcon> About
        </NavLink>
      </Typography>
      <Typography
        as="li"
        variant="small"
        color="blue-gray"
        className="p-1 font-medium"
      >
        <NavLink to="#features" className="flex items-center hover:text-blue-500 transition-colors">
         <Square3Stack3DIcon className="w-4 h-4 mr-1"></Square3Stack3DIcon>  Features
        </NavLink>
      </Typography>
      <Typography
        as="li"
        variant="small"
        color="blue-gray"
        className="p-1 font-medium"
      >
        <NavLink to="#pricing" className="flex items-center hover:text-blue-500 transition-colors">
         <CreditCardIcon className="w-4 h-4 mr-1"></CreditCardIcon> Pricing
        </NavLink>
      </Typography>
      <Typography
        as="li"
        variant="small"
        color="blue-gray"
        className="p-1 font-medium"
      >
        <NavLink to="#contact" className="flex items-center hover:text-blue-500 transition-colors">
         <PhoneIcon className="w-4 h-4 mr-1"></PhoneIcon> Contact Us
        </NavLink>
      </Typography>

      <Typography
        as="li"
        variant="small"
        color="blue-gray"
        className="p-1 font-medium"
      >
        <NavLink to="#" className="flex items-center hover:text-blue-500 transition-colors">
         <UserCircleIcon className="w-4 h-4 mr-1"></UserCircleIcon> Account
        </NavLink>
      </Typography>
    </ul>
  );
}
 
export default function NavBar() {
  const [openNav, setOpenNav] = React.useState(false);
 
  const handleWindowResize = () =>
    window.innerWidth >= 960 && setOpenNav(false);
 
  React.useEffect(() => {
    window.addEventListener("resize", handleWindowResize);
 
    return () => {
      window.removeEventListener("resize", handleWindowResize);
    };
  }, []);
 
  return (
    <Navbar variant="gradient" color="black" className="fixed z-10 w-full mx-auto px-6 py-3 rounded-none">
      <div className="flex items-center justify-between text-black">
        <Typography
          variant="h6"
          className="mr-4 cursor-pointer py-1.5"
        >
          Medical
        </Typography>
        <div className="hidden lg:block">
          <NavList />
        </div>
        <IconButton
          variant="text"
          className="ml-auto h-6 w-6 text-inherit hover:bg-transparent focus:bg-transparent active:bg-transparent lg:hidden"
          ripple={false}
          onClick={() => setOpenNav(!openNav)}
        >
          {openNav ? (
            <XMarkIcon className="h-6 w-6" strokeWidth={2} />
          ) : (
            <Bars3Icon className="h-6 w-6" strokeWidth={2} />
          )}
        </IconButton>
      </div>
      <Collapse open={openNav}>
        <NavList />
      </Collapse>
    </Navbar>
  );
}