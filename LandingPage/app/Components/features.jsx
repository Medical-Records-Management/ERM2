import React from 'react'
import { Typography } from '@material-tailwind/react'
import { 
    BookOpenIcon,
    CalculatorIcon,
    ChartPieIcon,
    Square3Stack3DIcon
} from "@heroicons/react/24/solid";



const features = [
    {
        name: "Patient Record Management", 
        decription:
            "Quis tellus eget adipiscing convallis sit sit eget aliquet quis. Suspendisse eget egestas a elementum pulvinar et feugiat blandit at.In mi viverra elit nunc.",
         icon:BookOpenIcon
    },
    {
        name: "Billing", 
        decription:
            "Quis tellus eget adipiscing convallis sit sit eget aliquet quis. Suspendisse eget egestas a elementum pulvinar et feugiat blandit at.In mi viverra elit nunc.",
         icon:CalculatorIcon
    },
    {
        name: "Reports and Analysis", 
        decription:
            "Quis tellus eget adipiscing convallis sit sit eget aliquet quis. Suspendisse eget egestas a elementum pulvinar et feugiat blandit at.In mi viverra elit nunc.",
         icon:ChartPieIcon
    },
    {
        name: "Stock Management", 
        decription:
            "Quis tellus eget adipiscing convallis sit sit eget aliquet quis. Suspendisse eget egestas a elementum pulvinar et feugiat blandit at.In mi viverra elit nunc.",
         icon:Square3Stack3DIcon
    }
]


export default function Features() {
  return (
    <div id='features' className='relative bg-white grid gap-16 min-h-[100vh] p-8 grid-cols-1 grid-rows-auto'>
        <div className='text-black min-h-[100px] flex flex-col gap-6 w-full justify-center place-items-center text-center'>
            <Typography  variant="h1" color="White" className="text-bold text-3xl">
                Everything you need to get Started
            </Typography>
            <span className="text-bold text-[18px] w-full md:w-2/4 ">
                Quis tellus eget adipiscing convallis sit sit eget aliquet quis. 
                Suspendisse eget egestas a elementum pulvinar et feugiat blandit at.
                 In mi viverra elit nunc.
            </span>

        </div>
        <div className='grid mx-auto px-10 gap-6 lg:grid-cols-2'>
            {
                features.map((feature)=>(
                    <div className='flex m-4'>
                        <h1 className='block p-4  text-white'>
                            <feature.icon className="w-7 h-7 p-1 bg-[#062D99] rounded " />
                        </h1>
                        <div className='m-4'>
                            <h3 className='text-black text-bold text-[20px]'>
                                {feature.name}
                            </h3>
                            <p className='text-gray-900 text-[15px]'>
                                {feature.decription}
                            </p>
                        </div>
                    </div>

                ))
            };
        </div>
        
    </div>
  )
}