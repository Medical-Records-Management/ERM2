import React from 'react'
import {
    Card,
    CardHeader,
    CardBody,
    CardFooter,
    Typography,
    Tooltip,
  } from "@material-tailwind/react";
import {BsFacebook, BsTwitter,BsInstagram} from "react-icons/bs"
const teams = [
    {
        name : "Progress Munoriarwa",
        position : "Co-founder / Developer",
        image : "./images/me.jpg",
    },
    {
        name : "Genius Chinembiri",
        position : "Co-founder / Developer",
        image : "./images/gc.jpg",
    },
    {
        name : "Grace Rabbit",
        position : "Senior Developer",
        image : "./images/tr.jpg",
    }
]

export default function Team() {
  return (
    <div className='grid grid-rows-auto min-h-[100vh]  p-4'>
        <div className='m-6 mx-auto text-center'>
            <Typography
             variant="h1" 
             color="white" 
             className='mb-6 text-white md:text-4xl lg:text-5xl'
             >
                Meet our team
            </Typography>
            <Typography
             variant="lead" 
             color="white" 
             className='mb-6 opacity-80'
             >
                We’re a dynamic group of individuals who are passionate about what we do.
            </Typography>
        </div>
        <div className='grid grid-cols-1 mx-auto gap-6 lg:grid-cols-3'>
            {
                teams.map((team)=>(
                    // <div className='bg-[#434757] flex flex-col h-[60vh] rounded lg: w-[25vw]'>
                    //     <div className='h-3/4 px-10 py-4'>
                    //         <img src={team.image} className='h-full w-full rounded-[100%] p-4' alt="" />
                    //     </div>
                    //     <div className='bg-yellow-50 text-center'>
                    //         <h3>{team.name}</h3>
                    //         <p>{team.position}</p>
                    //     </div>
                    // </div>

                    <Card className="w-80 bg-[#05386b] shadow-sm">
                        <CardHeader floated={false} className="h-60">
                            <img src={team.image} alt="profile-picture" />
                        </CardHeader>
                        <CardBody className="text-center">
                            <Typography variant="h4" color="white" className="mb-2 opacity-80">
                            {team.name}
                            </Typography>
                            <Typography className="font-medium text-[#db4ddb]" textGradient>
                            {team.position}
                            </Typography>
                        </CardBody>
                        <CardFooter className="flex justify-center gap-7 pt-2">
                            <Tooltip content="Like">
                            <Typography
                                as="a"
                                href="#facebook"
                                variant="lead"
                                textGradient
                            >
                                <BsFacebook className="text-[#070717]" />
                            </Typography>
                            </Tooltip>
                            <Tooltip content="Follow">
                            <Typography
                                as="a"
                                href="#twitter"
                                variant="lead"
                                textGradient
                            >
                                <BsTwitter className="text-blue-400" />
                            </Typography>
                            </Tooltip>
                            <Tooltip content="Follow">
                            <Typography
                                as="a"
                                href="#instagram"
                                variant="lead"
                                textGradient
                            >
                                <BsInstagram className="text-[purple]" />
                            </Typography>
                            </Tooltip>
                        </CardFooter>
                    </Card>
                ))
            }
        </div>
    </div>
  )
}