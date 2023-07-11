import React from 'react'
import { Typography , Button} from '@material-tailwind/react'
function Pricing(props) {

  return (
    <div id='pricing' className='relative grid gap-16 h-full p-8 grid-cols-1 grid-rows-auto'>
        <div className='text-white min-h-[100px] flex flex-col gap-6 w-full justify-center place-items-center text-center'>
            <Typography  variant="h1" color="White" className="text-bold text-3xl">
                Pricing plans for teams of all sizes
            </Typography>
            <span className="text-bold text-[18px] w-full md:w-2/4 ">
                Choose an affordable plan that’s packed with the best features for engaging your audience, 
                creating customer loyalty, and driving sales.
            </span>

        </div>
        <div className='grid gap-6 text-white m-auto lg:grid-cols-3'>
            <div className='flex flex-col border-[1px] border-gray-700 rounded-[30px] min-h-[70vh] md:min-h-[50vh] w-[50vh]  lg:min-h-[90vh] p-8 hover:w-70 cursor-pointe'>
                <span className='text-bold text-blue-700 text-[18px] lg:text-[18px] md:text-3xl'>
                    Basic Plan
                </span>
                <p className='mt-4 md:text-[18px]'>The essentials to provide your best work for clients</p>
                <p className=' flex mt-4 mb-6 place-items-baseline'>
                    <span className='text-bold text-5xl mr-2'>$ 10 /</span><span>month</span>
                </p>
                <p className='text-bold text-blue-700 mt-6 md:text-[20px] lg:text-sm '>What's Included</p>
                <ul role="list" className='mt-6 mb-8 grid grid-cols-1 gap-4 text-sm leading-6 md:text-[20px] lg:text-sm'>
                    <li className='flex gap-x-3'> 
                        <svg class="h-6 w-5 flex-none text-indigo-600" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clip-rule="evenodd" />
                        </svg>
                        Member resources
                    </li>
                    <li className='flex gap-x-3'> 
                        <svg class="h-6 w-5 flex-none text-indigo-600" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clip-rule="evenodd" />
                        </svg>
                        Member resources
                    </li>
                    <li className='flex gap-x-3'> 
                        <svg class="h-6 w-5 flex-none text-indigo-600" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clip-rule="evenodd" />
                        </svg>
                        Member resources
                    </li>
                    <li className='flex gap-x-3'> 
                        <svg class="h-6 w-5 flex-none text-indigo-600" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clip-rule="evenodd" />
                        </svg>
                        Member resources
                    </li>
                </ul>
                <button className='rounded w-full h-10 m-auto lg:h-10 bg-blue-700 md:h-20'>Start now</button>
            </div> 
            <div className='flex flex-col border-[1px] border-gray-700 rounded-[30px] min-h-[70vh] md:min-h-[50vh] w-[50vh]  lg:min-h-[90vh] p-8 hover:w-70 cursor-pointe'>
                <span className='text-bold text-blue-700 text-[18px] lg:text-[18px] md:text-3xl'>
                    Compact Plan
                </span>
                <p className='mt-4 md:text-[18px]'>The essentials to provide your best work for clients</p>
                <p className=' flex mt-4 mb-6 place-items-baseline'>
                    <span className='text-bold text-5xl mr-2'>$ 20 /</span><span>month</span>
                </p>
                <p className='text-bold text-blue-700 mt-6 md:text-[20px] lg:text-sm '>What's Included</p>
                <ul role="list" className='mt-6 mb-8 grid grid-cols-1 gap-4 text-sm leading-6 md:text-[20px] lg:text-sm'>
                    <li className='flex gap-x-3'> 
                        <svg class="h-6 w-5 flex-none text-indigo-600" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clip-rule="evenodd" />
                        </svg>
                        Member resources
                    </li>
                    <li className='flex gap-x-3'> 
                        <svg class="h-6 w-5 flex-none text-indigo-600" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clip-rule="evenodd" />
                        </svg>
                        Member resources
                    </li>
                    <li className='flex gap-x-3'> 
                        <svg class="h-6 w-5 flex-none text-indigo-600" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clip-rule="evenodd" />
                        </svg>
                        Member resources
                    </li>
                    <li className='flex gap-x-3'> 
                        <svg class="h-6 w-5 flex-none text-indigo-600" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clip-rule="evenodd" />
                        </svg>
                        Member resources
                    </li>
                </ul>
                <button className='rounded w-full h-10 m-auto lg:h-10 bg-blue-700 md:h-20'>Start now</button>
            </div> 
            <div className='flex flex-col border-[1px] border-gray-700 rounded-[30px] min-h-[70vh] md:min-h-[50vh] w-[50vh]  lg:min-h-[90vh] p-8 hover:w-70 cursor-pointe'>
                <span className='text-bold text-blue-700 text-[18px] lg:text-[18px] md:text-3xl'>
                    Premium Plan
                </span>
                <p className='mt-4 md:text-[18px]'>The essentials to provide your best work for clients</p>
                <p className=' flex mt-4 mb-6 place-items-baseline'>
                    <span className='text-bold text-5xl mr-2'>$ 30 /</span><span>month</span>
                </p>
                <p className='text-bold text-blue-700 mt-6 md:text-[20px] lg:text-sm '>What's Included</p>
                <ul role="list" className='mt-6 mb-8 grid grid-cols-1 gap-4 text-sm leading-6 md:text-[20px] lg:text-sm'>
                    <li className='flex gap-x-3'> 
                        <svg class="h-6 w-5 flex-none text-indigo-600" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clip-rule="evenodd" />
                        </svg>
                        Member resources
                    </li>
                    <li className='flex gap-x-3'> 
                        <svg class="h-6 w-5 flex-none text-indigo-600" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clip-rule="evenodd" />
                        </svg>
                        Member resources
                    </li>
                    <li className='flex gap-x-3'> 
                        <svg class="h-6 w-5 flex-none text-indigo-600" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clip-rule="evenodd" />
                        </svg>
                        Member resources
                    </li>
                    <li className='flex gap-x-3'> 
                        <svg class="h-6 w-5 flex-none text-indigo-600" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clip-rule="evenodd" />
                        </svg>
                        Member resources
                    </li>
                </ul>
                <button className='rounded w-full h-10 m-auto lg:h-10 bg-blue-700 md:h-20'>Start now</button>
            </div> 
        </div>
       
    </div>

    
  )
}

export default Pricing