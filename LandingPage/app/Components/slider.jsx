import React from 'react'
import { Carousel, Typography, Button} from '@material-tailwind/react'
import image_1 from '../images/1.jpg'
import image_2 from '../images/2.jpg'
import image_3 from '../images/3.jpg'

export default function Slider(){
  return (
    <Carousel className='h-[100vh]'>
        <div className='relative h-full w-full'>
            <img src="./images/1.jpg"
             alt="image 1" 
            className='h-full w-full object-cover' 
            />
            <div className='absolute inset-0 grid h-full w-full place-items-center bg-black/75'>
                <div className='w-3/4 text-center md:w-2/4'>
                    <Typography variant="h1" color="white" className="mb-4 md:text-4xl lg:text-5xl">
                        Medical Record Management
                    </Typography>
                    <Typography variant="lead" color="white" className="mb-12 opacity-80">
                        Lorem ipsum dolor sit, amet consectetur adipisicing elit. Eos libero fuga obcaecati natus
                            facere! Fugit amet in maxime iure dignissimos obcaecati sit assumenda eius dolor? Cumque 
                            fugiat quod explicabo voluptates!
                    </Typography>
                    <div className='flex justify-center gap-2'>
                        <Button size='lg' color='white' >
                            Get Started
                        </Button>
                    </div>
                </div>
            </div>
        </div>
        <div className='relative h-full w-full'>
            <img src="./images/2.jpg" alt="" 
            className='h-full w-full object-cover' />
            <div className='absolute inset-0 grid h-full w-full place-items-center bg-black/75'>
                <div className='w-3/4 text-center md:w-2/4'>
                    <Typography variant="h1" color="white" className="mb-4 md:text-4xl lg:text-5xl">
                        The Beauty of Nature
                    </Typography>
                    <Typography variant="lead" color="white" className="mb-12 opacity-80">
                        Lorem ipsum dolor sit, amet consectetur adipisicing elit. Eos libero fuga obcaecati natus
                            facere! Fugit amet in maxime iure dignissimos obcaecati sit assumenda eius dolor? Cumque 
                            fugiat quod explicabo voluptates!
                    </Typography>
                    <div className='flex justify-center gap-2'>
                        <Button size='lg' color='white' >
                            Get Started
                        </Button>
                    </div>
                </div>
            </div>
        </div>
        <div className='relative h-full w-full'>
            <img src="./images/3.jpg" alt="" className='h-full w-full object-cover' />
            <div className='absolute inset-0 grid h-full w-full place-items-center bg-black/75'>
                <div className='w-3/4 text-center md:w-2/4'>
                    <Typography variant="h1" color="white" className="mb-4 md:text-4xl lg:text-5xl">
                        The Beauty of Nature
                    </Typography>
                    <Typography variant="lead" color="white" className="mb-12 opacity-80">
                        Lorem ipsum dolor sit, amet consectetur adipisicing elit. Eos libero fuga obcaecati natus
                            facere! Fugit amet in maxime iure dignissimos obcaecati sit assumenda eius dolor? Cumque 
                            fugiat quod explicabo voluptates!
                    </Typography>
                    <div className='flex justify-center gap-2'>
                        <Button size='lg' color='white' >
                            Get Started
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    </Carousel>
  )
}