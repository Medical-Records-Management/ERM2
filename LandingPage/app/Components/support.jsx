import React from 'react'
import { Typography } from '@material-tailwind/react'
import { 
    BugAntIcon,
    ChatBubbleBottomCenterIcon,
    ComputerDesktopIcon,
    PhoneIcon,
} from '@heroicons/react/24/outline'


const contacts = [
    {
        name : "Sales", 
        dectription : "some sales stuff that should come here. Details will folow later",
        cell : "+263 771 111 111",
        icon:ChatBubbleBottomCenterIcon
    },
    {
        name : "Bug Report", 
        dectription : "some Bug stuff that should come here. Details will folow later",
        cell : "+263 771 111 111",
        icon:BugAntIcon
    },
    {
        name : "Tech Support", 
        dectription : "some sales tech that should come here. Details will folow later",
        cell : "+263 771 111 111",
        icon:ComputerDesktopIcon
    }
]

export default function Support() {
  return (
    <div id='contact' className='bg-white grid grid-rows-auto min-h-[100vh]  p-4'>
        <div className='m-6 mx-auto'>
            <Typography
             variant="h1" 
             color="white" 
             className='mb-6 text-[#05386b] md:text-4xl lg:text-5xl'
             >
                Support Center
            </Typography>
{/* 
            <ul>
                <li className='flex m-4'>
                    <CheckCircleIcon className='w-6 h-6 text-green-600 pr-2'></CheckCircleIcon>
                    24 hr Support
                </li>
                <li className='flex m-4'>
                    <CheckCircleIcon className='w-6 h-6 text-green-600 pr-2'></CheckCircleIcon>
                    On-site Support
                </li>
                <li className='flex m-4'>
                    <CheckCircleIcon className='w-6 h-6 text-green-600 pr-2'></CheckCircleIcon>
                    Fast response time
                </li>
            </ul> */}
        </div>
        <div className='grid grid-cols-1 gap-6 mx-auto lg:p-4'>
            {
                contacts.map((contact)=>(
                    
                    <div className='flex gap-6 mb-8'>
                        <div>
                            <contact.icon className="text-white w-10 h-10  bg-[#05386b] p-2 rounded" />
                        </div>
                        <div className='flex flex-col'>
                            <Typography
                                variant="lead" 
                                className="text-[#5cbdb9] text-bold" 
                            >
                                {contact.name}

                             </Typography>
                            <p className='opacity-80'>{contact.dectription}</p>
                            <p className='flex mt-2'>
                                <PhoneIcon className='w5 h-5 pr-1'></PhoneIcon> 
                                {contact.cell}
                            </p>
                        </div>
                    </div>

                ))
            }
        </div>
    </div>
  )
}